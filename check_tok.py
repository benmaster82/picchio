import json
t = json.load(open('D:/gptoss_i4/tokenizer.json', encoding='utf-8'))
v = t.get('model', {}).get('vocab', {})
m = t.get('model', {}).get('merges', [])
print(f"Vocab entries: {len(v)}")
print(f"Merges: {len(m)}")
print(f"Top-level keys: {list(t.keys())}")
print(f"Model keys: {list(t.get('model', {}).keys())}")
# Sample entries
items = list(v.items())
print(f"First 3: {items[:3]}")
print(f"Last 3: {items[-3:]}")
# Check added_tokens
added = t.get('added_tokens', [])
print(f"Added tokens: {len(added)}")
if added:
    for a in added[:5]:
        print(f"  id={a.get('id')} content={repr(a.get('content'))}")
