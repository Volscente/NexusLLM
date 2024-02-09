# Transformers
## Resources
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)

## Input
The input of a Transformer, and in general of each Neural Network, is a Tensor. In this case, it is obtained from the input
text through an **Embedding Algorithm**.

## Architecture
The basic architecture of a Transformer is composed by **Encoders** and **Decoders**.

### Encoder
It is composed by a **Self-Attention** layer and a **Feed Forward Neural Network**.
![Encoder Architecture](./images/encoder.png)

### Decoder
It has a similar architecture that an encoder, but with an additional layer in the middle to
help focus on relevant part of the input sentence.
![Encoder Architecture](./images/decoder.png)
