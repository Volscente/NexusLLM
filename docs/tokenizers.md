# Introduction
## Resources
### YouTube
- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
### Playground
- [TikTokenizer](https://tiktokenizer.vercel.app/)
### Libraries
- [TikToken](https://github.com/openai/tiktoken)
- [SentencePiece](https://github.com/google/sentencepiece)

## Definition
It is used the main element while using LLMs, and it converts strings into vectors.

## Strings and Unicode in Python
In Python you can retrieve the Unicode value of a character through `ord('<char>)`.

The Python interpreter does not therefore see characters, but those numbers instead.

UTF-8/16/32 are specific encoding of Unicode: they define how to represent Unicode characeters in bytes.
For example: `'hello'.encode('utf-8')` would return the bytes for those 5 characters.

```python
list('Hello'.encode('utf-8'))# [72, 101, 108, 108, 111] -> List of raw bytes
```

UTF-8 uses dynamic number of bytes up to 4, while UTF-16 and UTF-32 use always the same amount of 4 bytes, 
leading to many 0s in their encodings, especially for single english characters.

It would be amazing to directly feed bytes into an LLM, but the context window would be too high! Therefore, we still
need the tokenization process.


## Common Problems
### Diluted Tokens
The reason why LLMs work in different ways between different languages and/or topics,
is because of the Tokenizer (most of the time).

This can be due to different reasons, but if we have a look on how GPT2 Tokenizer behaves
with Korean with respect to english, we can see an interesting pattern. The tokens required
to tokenize a Korean text are much more than the ones required to tokenize a similar 
english text. This is because the tokenizer has been trained on english text mainly.

The effect is that the output tokens for the Korean text are much more sparse, and therefore
the LLM does not work very well.

### Whitespace Character
One key component while tokenizing code text is how to tokenize space. GPT2 use a single token for
each space, leading to a Diluted Tokens problem.

GPT4o Tokenizer (cl100k_base) improved this behaviour by using less tokens for the indentation.

# Characteristics
## Tokens Vocabulary
It represents the tokens space through which the tokenizer can convert the text.
The bigger is this vocabulary, the lesser tokens would be required to tokenize a text.

More tokens in the vocabulary means that the tokenizer would be able to better represent the
meaning of a text, fighting the "Diluted Tokens" problem.

However, too big vocabulary might influence the LLM softmax function for the token sampling
while constructing its output.

## Splitting
The very first step of any Encoder is the text split.

GPT-2, for example, does that through a REGEX, which can work differently between lowercase and uppercase.
Also, the way in which the sentence is separated before encoding it can affect the final result.

There are also rules when need to split text that includes, for example, code.

This is the main change between GPT-2 and GPT-4 Tokenizers.

# Techniques
## Byte Pair Encoding
It is a technique that want to compress the output encoding by encoding together the pairs of most common bytes.

For example: [aaabbaabaacaa] &rarr; The sequence "aa" is the most common &rarr; Z = aa &rarr; [ZabbZbZcZ]   

This would reduce the length of the output sequence. This can be done recursively and shortening each time the output sequence. 

It is possible to perform a hyperparameter tuning process in order to understand which is the best Vocabulary size that has the
best compression (i.e, the number of times we repeat the Byte Pair Encoding).

## Special Tokens
With the `TikToken` library it is possible to add special tokens to the Encoder.

For example the `<endoftext>` and assign it with an unused index.

# Training
## General
The Tokenizer has its own training set, separated from the LLM's training.

Taking into account the training dataset and the "Diluted Tokens" problem, it becomes clear that, the more words in the Tokenizer
sees in the training dataset that are, for example, in Japanese, the better the Tokenizer would group up these words into 
the same token. In this way, it would represent a Japanese sentence with far less tokens.

# Libraries
## SentencePiece
Unlike TikToken, SentencePiece is pretty good with both training and inference:
- TikToken - Encode to UTF-8 and then applied BPE
- SentencePiece - Apply BPE directly on the code points (and eventually falls back to UTF-8 for certain code points)