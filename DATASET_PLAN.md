# Persona 20M dataset plan

## Current local starter corpus
`data/train_20m.txt` is a synthetic, original starter corpus generated for this project. It is about 29.8M UTF-8 bytes and contains about 145k conversational/instruction examples plus short narrative passages.

This is intentionally not presented as a substitute for a large natural-language corpus. It is a bootstrap dataset so Persona 20M has much more language structure than the previous 24K-character corpus.

## Target
- Phase 1: 30M bytes — **starter corpus complete**
- Phase 2: 100M tokens/bytes — public pretraining data + original data
- Phase 3: 500M+ tokens — larger mixture with deduplication and quality filtering
- Later: add a held-out evaluation set and owner feedback data separately

## Public-data route
`prepare_public_data.py` can stream the public TinyStories dataset from Hugging Face without an API key. TinyStories is listed as CDLA-Sharing-1.0 on its dataset card. Review the license before redistributing data or a trained model.

For Persona's conversational goal, public text pretraining should be followed by a separate instruction/conversation fine-tuning stage rather than treating stories alone as chat training.
