# Character Based LLM
## Definition
This is an exercise from Andrej Karpathy that aims to demonstrate how LLMs works.
The code implements a LLM based on character prediction: given a set of characters and a context, it will try to predict
the very next character of the sequence.
# Large Language Model (LLM)
## Definition
It is a Language Model because it models a sequence of words/characters in a probabilistic way: 
probabilistically predict what word/character will follow the previous one. 
It starts from the given prompt and then tries to complete the sequence of it, generating the output.
## Neural Network Architecture
### Resources
- The reference paper is called [Attention is All You Need](https://arxiv.org/abs/1706.03762).
- This [video](https://www.youtube.com/watch?v=kCc8FmEb1nY) from Andrej Karpathy explains how LLMs work
### Transformer Architecture
It is the Neural Network technology behind the Large Language Models like ChatGPT (Chat Generally Pre-Trained Transformer).
### Tokenization
It is an important data preprocessing operation which converts the single portion of the sequence 
(characters or tokens of words) into numerical value based on all the possible values of the train vocabulary.
Some of the most famous Tokenizer are:
- [Google SentencePiece](https://github.com/google/sentencepiece)
- [TikToken from OpenAI](https://github.com/openai/tiktoken)
