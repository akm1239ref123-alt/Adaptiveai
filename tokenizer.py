import json
from pathlib import Path

class ByteTokenizer:
    """Simple lossless byte-level tokenizer: exactly 256 tokens."""
    vocab_size = 256

    def encode(self, text: str):
        return list(text.encode('utf-8', errors='replace'))

    def decode(self, ids):
        return bytes(int(i) % 256 for i in ids).decode('utf-8', errors='replace')

    def save(self, path):
        Path(path).write_text(json.dumps({'type':'byte','vocab_size':256}), encoding='utf-8')

    @classmethod
    def load(cls, path):
        return cls()
