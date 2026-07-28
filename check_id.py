import json, struct
# Read from binary vocab
with open('D:/gptoss_i4/picchio_vocab.bin', 'rb') as f:
    magic = f.read(4)
    total, n_added, eos_id, pad_id = struct.unpack('<IIII', f.read(16))
    tokens = []
    for i in range(total):
        l = struct.unpack('<H', f.read(2))[0]
        s = f.read(l).decode('utf-8', errors='replace')
        tokens.append(s)

for tid in [284, 200005, 200006, 200007, 200008, 200002, 200012]:
    if tid < len(tokens):
        print(f"  {tid} = {repr(tokens[tid])}")
    else:
        print(f"  {tid} = (fuori range)")
