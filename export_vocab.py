#!/usr/bin/env python3
"""export_vocab.py — Converte tokenizer.json HF in formato binario per Picchio.

Formato output (picchio_vocab.bin):
  Header:
    4 byte: magic "PVOC"
    4 byte: vocab_size (uint32 LE)
    4 byte: n_added (uint32 LE) 
    4 byte: eos_id (uint32 LE)
    4 byte: pad_id (uint32 LE)
  Per ogni token (0..vocab_size+n_added-1):
    2 byte: lunghezza stringa (uint16 LE)
    N byte: stringa UTF-8 (raw bytes, NO null terminator)

Questo formato è ~4 MB e si carica in <50ms in C.
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
    
    # Costruisci la mappa completa id → bytes
    max_id = max(vocab.values()) if vocab else 0
    for at in added_tokens:
        if at['id'] > max_id:
            max_id = at['id']
    
    total = max_id + 1
    tokens = [''] * total
    
    # Vocabolario base
    for token_str, token_id in vocab.items():
        tokens[token_id] = token_str
    
    # Added tokens (special)
    for at in added_tokens:
        tokens[at['id']] = at.get('content', '')
    
    # Trova EOS e PAD
    eos_id = 200002  # <|return|>
    pad_id = 199999  # <|endoftext|>
    for at in added_tokens:
        if at.get('content') == '<|return|>':
            eos_id = at['id']
        elif at.get('content') == '<|endoftext|>':
            pad_id = at['id']
    
    # Scrivi formato binario
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
    print(f"✓ {total} token esportati in {output_path} ({out_size/1024/1024:.1f} MB)")
    print(f"  eos_id={eos_id} pad_id={pad_id}")
    
    # Verifica
    print(f"  token[0] = {repr(tokens[0])}")
    print(f"  token[100] = {repr(tokens[100])}")
    print(f"  token[200002] = {repr(tokens[200002])}")


if __name__ == '__main__':
    tok_path = sys.argv[1] if len(sys.argv) > 1 else 'D:/gptoss_i4/tokenizer.json'
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'D:/gptoss_i4/picchio_vocab.bin'
    export_vocab(tok_path, out_path)
