# Introduction
## Resources
### YouTube
- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE)
### Playground
- [TikTokenizer](https://tiktokenizer.vercel.app/)

## Definition
It is used the main element while using LLMs, and it converts strings into vectors.

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