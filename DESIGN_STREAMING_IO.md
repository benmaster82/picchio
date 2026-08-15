# Streaming I/O roadmap: aligned expert store and OS-bypass

Status: design proposal (not yet implemented). This document specifies a fast
storage path for Picchio's expert streaming and the roadmap that leads to a full
OS-bypass data path. It is deliberately incremental: each phase is useful on its
own, is measured before the next is started, and keeps the current safetensors
path working as a fallback.

## 0. What actually costs time (and what does not)

Picchio is bandwidth and IOPS bound, not syscall bound. Being precise about the
overhead is what keeps this roadmap honest:

- **Syscall count is negligible.** A `pread` / `ReadFile` costs about 1 to 5 us.
  Qwen streams ~384 expert reads per token, so ~1 ms/token of syscall overhead
  against ~1250 ms/token at 0.8 tok/s. That is under 0.2 percent. Removing
  syscalls is not the goal.
- **Path resolution is negligible.** Handles are opened once and kept open; there
  is no per-read path walk.
- **The page cache is a real cost.** Every buffered read copies disk to page cache
  to user buffer, and the page cache competes with Picchio's own LRU. This is the
  measured PILOT problem in `DESIGN.md` section 0.8: the OS cache evicts pages and
  the same bytes are read twice.
- **Queue depth 1 is a real cost.** Synchronous `pread` keeps one read in flight
  per thread. An NVMe reaches full bandwidth only at high queue depth (16 to 128).
  With `IO_THREADS=4` the effective depth is ~4. On small reads (the 2.65 MB Qwen
  experts) this leaves a 5 to 10x factor of IOPS on the table.

Two consequences frame everything below:

1. **Large experts already saturate the disk.** `DESIGN.md` section 0.11 measured
   the ~12 MB GPT-OSS experts reading at the NVMe sequential ceiling (~494 MB/s).
   There is little overhead left to remove there. The wins in this document scale
   with how small and how numerous the reads are, so they matter most for the
   many-small-experts regime: Qwen3 today, and the industry trend toward more and
   smaller experts (DeepSeek, Kimi, later Qwen).
2. **Nothing here beats fast hardware.** On a USB bridge (single queue, high
   latency) these techniques give little, because the bridge is the bottleneck.
   The prerequisite is an internal Gen3+ NVMe. On Gen4/5 the disk stops being the
   wall and compute or overhead becomes the limiter, which is exactly where the
   later phases pay off.

## 1. Phase S1: flat aligned expert store (`.picchioflat`)

Turn "open, find tensor, walk extents, read" into "one aligned DMA at a known
LBA". A purpose-built layout where every expert is block-aligned and contiguous,
with a flat resident index. It is the "file system for the job", but in its
pragmatic form: first as a single file on a normal file system (about 90 percent
of the benefit at minimal risk), later as a raw partition (phase S4).

### 1.1 On-disk format

Little-endian. `BS` is the alignment block size (4096 by default: matches the
NVMe logical block and the memory page, so it is legal for unbuffered/O_DIRECT
reads).

```
[ Superblock ]  (one BS block at offset 0)
  u64  magic          "PCHIOFL1"
  u32  version
  u32  block_size     BS (4096)
  u32  hidden, n_layers, n_experts, topk, moe_inter   (arch echo, sanity check)
  u32  n_experts_total    (= n_layers * n_experts)
  u64  index_offset       (BS-aligned start of the expert index)
  u64  index_len
  u64  data_offset        (BS-aligned start of the payload region)
  u8   sha256_index[32]   (hash of the index blob; ties into the L0 contract)
  ... (reserved, pad to BS)

[ Expert index ]  (array of n_experts_total entries, BS-aligned start)
  struct ExpertLoc {          // 24 bytes
    u64 offset;               // byte offset of this expert, multiple of BS
    u32 len;                  // exact payload bytes (unpadded), for slicing
    u32 padded_len;           // len rounded up to BS, the bytes to read
    u64 hash;                 // truncated sha256 of the payload (spot-check)
  }
  // addressed by index[layer * n_experts + eid]

[ Payload region ]  (each expert BS-aligned and contiguous)
  For each expert, in (layer, eid) order:
    gate_up_proj packed (INT4)
    gate_up_proj scales (f32, group-scaled 64)
    down_proj    packed (INT4)
    down_proj    scales (f32)
    gate_up bias (f32, if present)   down bias (f32, if present)
    (pad to a multiple of BS so the next expert starts aligned)
```

The sub-tensor sizes are fully determined by `config.json` (`moe_inter`, `hidden`,
group size), so the runtime slices the single expert buffer without per-sub-tensor
offsets. This is the exact "one read equals one expert" property, now aligned and
fragmentation-free. The padding waste is under one block per expert (about 2 KB
average) against a 2.65 MB expert: negligible.

### 1.2 Index size

`n_experts_total * 24` bytes, fully resident. Qwen3-30B: 48 * 128 = 6144 experts,
about 144 KB. Loaded once at startup. O(1) lookup, no disk access on the hot path.

### 1.3 Dense weights

v1 is experts-only: the dense part (attention F32, embed/head INT8, norms, router)
stays in the existing safetensors and is loaded once at startup with large
sequential reads, off the hot path. A later variant can place an aligned dense
blob in the same file (`dense_offset`/`dense_len` reserved in the superblock),
which becomes useful for the raw-partition phase.

### 1.4 Producer

A converter post-pass (or a new writer in `convert.py` / `convert_streaming*.py`)
walks the converted INT4 experts in `(layer, eid)` order, writes each expert's
sub-tensors contiguously, pads to `BS`, and records `offset/len/padded_len/hash`.
Picchio's experts are already contiguous in the current container, so this is a
repack, not a re-quantization. It runs from the converted model, no re-download.

### 1.5 Consumer (interception point)

The natural hook is `expert_load` / `cache_load_batch` in `picchio.c` (backed by
`st.h`). If a `.picchioflat` file is present next to the model, the expert backend
switches to: `loc = index[layer*n_experts + eid]; read(fd, aligned_buf,
loc.padded_len, loc.offset)`, then slice. No `st_find`, no path walk. If the flat
file is absent, the current safetensors path is used unchanged. The flat store is
an optional fast lane produced by one extra convert step.

## 2. Phase S2: unbuffered DMA (`O_DIRECT` / `FILE_FLAG_NO_BUFFERING`)

Read straight into the LRU slot buffers, bypassing the page cache. Requires the
buffer, the file offset, and the length to be `BS`-aligned, which S1 already
guarantees (that is why S1 comes first). Removes the disk-to-page-cache copy and,
more importantly, ends the contention between the OS page cache and Picchio's LRU.
This is expected to recover the PILOT regression (about 11 percent in
`DESIGN.md` section 0.8) and to make the miss count the clean, deterministic
metric it should be. Low complexity once S1 exists. Portable (POSIX `O_DIRECT`,
Windows `FILE_FLAG_NO_BUFFERING`).

## 3. Phase S3: async high queue depth (`io_uring` / IOCP)

Replace synchronous per-thread reads with asynchronous submission. Picchio already
computes the set of experts a layer needs (the batch union in the prefill path,
and the top-k in decode), so it can submit all of a layer's reads at once and reap
them as they complete. This lifts the effective queue depth from ~4 to the size of
the batch, which is where small-read NVMe throughput actually lives.

- Linux: `io_uring` (submit N reads, one syscall, reap completions). Pure C, no
  external dependency.
- Windows: overlapped I/O plus an IOCP completion port. Also dependency-free.

This is the largest realistic win for the Qwen-style regime, and it composes with
S1 (aligned) and S2 (unbuffered). It stays inside the kernel: it is not literal
OS-bypass, but it captures most of the "anti-OS" benefit while remaining
cross-platform.

## 4. Phase S4: raw partition (no file system)

Place the `.picchioflat` image on a dedicated raw partition or block device and
address experts by LBA directly. The file system is now fully out of the data
path: no extents, no metadata, no fragmentation, guaranteed alignment. The format
from S1 is unchanged; only the backing store changes (a block device instead of a
file). Moderate effort, and it is the stepping stone to S5. Cross-platform in
principle (open the raw device), though it wants a disk you can dedicate to the
model.

## 5. Phase S5: user-space NVMe driver (SPDK class)

The moonshot. Unbind the NVMe from the kernel driver, map its submission and
completion queues into user space, and DMA experts straight from the SSD into the
compute buffers. Zero syscalls and zero kernel in the data path: minimum latency,
effectively unbounded queue depth, full control of the I/O scheduler.

Honest constraints:

- **Platform.** Realistically Linux, via VFIO/UIO, hugepages, and root. A
  user-space NVMe path on Windows is a research topic, not a product path.
- **Zero-dependency tension.** SPDK itself is a large dependency, which contradicts
  Picchio's ethos. A from-scratch pure-C driver is possible (admin and I/O queues,
  PRP lists, doorbells, MSI-X) but is a serious, device-specific project on its
  own.
- **Dedicated device.** In user space you take over the whole controller: no shared
  file system on that NVMe. The model gets its own disk.
- **ROI gate.** This only beats S1 plus S2 plus S3 when the disk is fast enough
  (Gen4/5) that overhead and latency, not raw bandwidth, are the limiter, and when
  the reads are small and numerous. That is a real future regime, but it is not
  today's hardware.

S5 is best treated as a **separate research track / linked project** (see below),
not part of the core engine, precisely because it is Linux-only, dedicates the
device, and pulls in the "own the hardware" complexity that the rest of Picchio
deliberately avoids.

## 6. Sequencing and ROI

| Phase | Technique | Main win | Effort | Portable | When it pays |
|---|---|---|---|---|---|
| S1 | flat aligned `.picchioflat` store | one aligned read per expert, no fragmentation | medium | yes | always (enables S2/S3) |
| S2 | unbuffered DMA (O_DIRECT / no-buffering) | no page-cache copy or LRU contention | low | yes | always; recovers the PILOT loss |
| S3 | async high queue depth (io_uring / IOCP) | saturate NVMe on small reads | medium | yes | many-small-experts (Qwen, DeepSeek) |
| S4 | raw partition | file system fully out of the path | medium | mostly | dedicated disk available |
| S5 | user-space NVMe driver | zero kernel in the data path | very high | Linux | Gen4/5 NVMe, small reads, research |

Recommended order: S1, then S2, then S3, measuring at each step. S1 to S3 are high
ROI and portable. S4 and especially S5 are for a dedicated-disk, fast-NVMe future
and are optional.

**Precondition for all of it:** put the model on an internal Gen3+ NVMe. On a USB
bridge these phases give little, because the bridge, not the OS, is the wall.

## 7. Linked projects and future tiers

Picchio's memory hierarchy (`DESIGN.md` section 2.1) is VRAM, then RAM, then disk.
This roadmap sharpens the **disk tier** (aligned, unbuffered, async, and finally
OS-bypass). Two adjacent tiers are best kept as separate but linked projects so the
core engine stays small and dependency-free:

- **Network / peer tier (below disk): Lumabri.** By the same author as Colibri, a
  pure-C P2P system that streams MoE experts from a swarm of peers instead of local
  disk: only dense, router, and KV stay local, routed experts run on remote peers
  with about 4 KB of activations crossing the wire, and touched bytes persist in a
  local sparse mirror for later reuse. It solves the tier below "the model does not
  fit in RAM", namely "the model does not fit on the local disk" (the exact 61 GB
  on a 120 GB drive problem). Its `LD_PRELOAD` shim intercepts file reads
  transparently, so on Linux a `.picchioflat`-backed Picchio could in principle be
  fed from peers without touching the core; the interception point is the same
  `expert_load` boundary as S1. Integrity via SHA256 blocks and ed25519 signing,
  plus spot-check replay of peer compute, mirrors Picchio's L0 manifest philosophy
  and the per-expert `hash` field in the S1 index.
- **User-space NVMe driver (phase S5): separate research track.** As argued above,
  the SPDK-class path is Linux-only, dedicates the device, and is a project of its
  own. Keeping it outside the core engine preserves the "pure C, zero external
  dependencies, runs anywhere" contract while leaving the door open for a
  specialized high-end build.

The through-line: the same expert-load boundary that S1 formalizes is where every
future tier plugs in (aligned disk, raw device, user-space NVMe, or a peer swarm),
so the core forward pass never has to change.
