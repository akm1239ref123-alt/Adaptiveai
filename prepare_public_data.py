"""Optional public-data downloader/preparer.

TinyStories is used only as a language-pretraining source. The dataset card lists
CDLA-Sharing-1.0. Review that license before redistributing a trained model.
This script requires `pip install datasets` and does not require an API key.
"""
from pathlib import Path
import argparse

p=argparse.ArgumentParser()
p.add_argument('--target-tokens',type=int,default=100_000_000)
p.add_argument('--output',default='data/public_pretrain.txt')
a=p.parse_args()
try:
    from datasets import load_dataset
except ImportError:
    raise SystemExit('Install the public-data helper dependency first: pip install datasets')

out=Path(__file__).parent/a.output
out.parent.mkdir(parents=True,exist_ok=True)
written=0
# Streaming avoids downloading the whole dataset into RAM.
ds=load_dataset('roneneldan/TinyStories', split='train', streaming=True)
with out.open('w',encoding='utf-8') as f:
    for row in ds:
        text=str(row.get('text','')).strip()
        if not text: continue
        f.write(text+'\n\n')
        written += len(text.encode('utf-8'))
        if written >= a.target_tokens:
            break
print(f'wrote {written:,} UTF-8 bytes to {out}')
print('Note: bytes are an approximate token count because Persona uses byte-level tokenization.')
