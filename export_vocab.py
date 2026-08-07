#!/usr/bin/env python3
"""export_vocab.py — Convert an HF tokenizer.json into Picchio's binary format.

Output format (picchio_vocab.bin):
  Header:
    4 bytes: magic "PVOC"
    4 bytes: vocab_size (uint32 LE)
    4 bytes: n_added (uint32 LE)
    4 bytes: eos_id (uint32 LE)
    4 bytes: pad_id (uint32 LE)
  For each token (0..vocab_size+n_added-1):
    2 bytes: string length (uint16 LE)
    N bytes: UTF-8 string (raw bytes, NO null terminator)

This format is ~4 MB and loads in <50ms in C.
"""

import json
import struct
import sys
from pathlib import Path


def export_vocab(tokenizer_path: str, output_path: str):
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tok = json.load(f)
    
    vocab = tok.get('model', {}).get('vocab', {})
    added_tokens = tok.get('added_tokens', [])
    
    # Build the complete id → bytes map
    max_id = max(vocab.values()) if vocab else 0
    for at in added_tokens:
        if at['id'] > max_id:
            max_id = at['id']

    total = max_id + 1
    tokens = [''] * total

    # Base vocabulary
    for token_str, token_id in vocab.items():
        tokens[token_id] = token_str

    # Added tokens (special)
    for at in added_tokens:
        tokens[at['id']] = at.get('content', '')

    # Find EOS and PAD
    eos_id = 200002  # <|return|>
    pad_id = 199999  # <|endoftext|>
    for at in added_tokens:
        if at.get('content') == '<|return|>':
            eos_id = at['id']
        elif at.get('content') == '<|endoftext|>':
            pad_id = at['id']
    
    # Write the binary format
    with open(output_path, 'wb') as f:
        # Header
        f.write(b'PVOC')
        f.write(struct.pack('<I', total))
        f.write(struct.pack('<I', len(added_tokens)))
        f.write(struct.pack('<I', eos_id))
        f.write(struct.pack('<I', pad_id))
        
        # Tokens
        for i in range(total):
            s = tokens[i].encode('utf-8')
            f.write(struct.pack('<H', len(s)))
            f.write(s)
    
    out_size = Path(output_path).stat().st_size
    print(f"✓ {total} tokens exported to {output_path} ({out_size/1024/1024:.1f} MB)")
    print(f"  eos_id={eos_id} pad_id={pad_id}")

    # Verify
    print(f"  token[0] = {repr(tokens[0])}")
    print(f"  token[100] = {repr(tokens[100])}")
    print(f"  token[200002] = {repr(tokens[200002])}")


if __name__ == '__main__':
    tok_path = sys.argv[1] if len(sys.argv) > 1 else 'D:/gptoss_i4/tokenizer.json'
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'D:/gptoss_i4/picchio_vocab.bin'
    export_vocab(tok_path, out_path)
