# Persona 20M

Target architecture:

- 256 byte vocabulary
- 512-token context
- 448 hidden dimensions
- 8 attention heads
- 8 Transformer blocks
- SwiGLU feed-forward blocks
- tied input/output embeddings

This configuration is approximately **19.68 million trainable parameters**, which is the intended ~20M milestone.

The model remains fully from-scratch: random initialization, local Transformer implementation, local tokenizer, local training loop. No OpenAI/Gemini/Vercel AI model is used for inference.
