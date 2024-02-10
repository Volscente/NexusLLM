# Introduction
## Resources
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)

# Architecture
The basic architecture of a Transformer is composed by **Encoders** and **Decoders**.

## Input
The input of a Transformer, and in general of each Neural Network, is a Tensor. In this case, it is obtained from the input
text through an **Embedding Algorithm**.

The embedded input is also called as *Tokens* and its length depends on the Transformer architecture.

Usually, if the input text is not long enough, **Padding** is used to fill the missing tokens.

It is also important to know that there are *Special* tokens used by the Neural Network to mark:
- [CLS] or 101 - Start of the sentence
- [SEP] or 102 - Separator of sentences
- [MASK] or 103 - Mask token for MLM (Masked Language Model)

## Encoder
It is composed by a **Self-Attention** layer and a **Feed Forward Neural Network**.
![Encoder Architecture](./images/encoder.png)

## Decoder
It has a similar architecture that an encoder, but with an additional layer in the middle to
help focus on relevant part of the input sentence.
![Encoder Architecture](./images/decoder.png)

# Self-Attention
## Definition
It is a mechanism that allows the model to look at other positions in the input sequence to get a 
better understanding of token.

Consider the sentence *"The animal didn't cross the street because it was too tired"*

The word *"it"* refers to the *"Animal"*:
![Self-Attention Example](./images/self_attention.png)

## Aggregated Weighted Context
It is one of the first for of *Self-Attention*.

Consider the sequence: `[18, 47, 56, 57, 58,  1, 15, 47, 58]`

While training the token `56`, the algorithm should retrieve its context ‘[18, 47]’ in order to learn the next token ‘57’
. Passing the whole context everytime is expensive. That’s why in Transformer architecture, it’s better to pass a more concise representation of the previous context: an aggregated weighted context. A very simplified version adopts just the average:
Token: ‘56’
Context: [18, 47] → ‘41.5’ (average)
Target: ‘57’


