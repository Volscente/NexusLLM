# General
## Alignment Process
### Definition
Usually, a LLM goes through the following two training steps:
- Pre-training
- Fine-Tuning

The *Alignment* process is used in order to collect feedbacks of users regarding a particular LLM's output 
and decide which is better.

### Example
1. Pre-Training phase
2. Fine-Tuning phase &rarr; *"Is pineapple on Pizza a Crime?"* &rarr; *"Putting pineapple on a Pizza violates the Geneva convention etc."*
2. Alignment phase &rarr; I

## Prompting
### Definition
Usually called also "Prompt Engineering", it is the process of crafting a good prompt for the desired purpose.

### Types
- **Zero-Shot Prompting** - Ask the LLM to solve a problem
- **Few-Shot Prompting** - Provide few examples on how to solve the problem and then ask
- **Chain-of-Thought Prompting** - Guide the LLM through the entire process it has to do to solve the problem

## Inference
### Sampling Techniques
How to choose which is the best output token?
1. Greedy Search - Most probable token
2. Random Sampling - Selects the next token according to the probability distribution, where each token is sampled 
proportionally to its predicted probability.
3. Temperature Sampling - Adjusts the probability distribution by a temperature parameter. 
Higher temperatures promote diversity, lower temperatures favor high-probability tokens.
4. Top-K sampling - Randomly samples from the top K most probable tokens.
5. 

### Performance
There are several ways to make the inference process more performing:
1. Quantisation - It uses lower precision memory in order to not lose many
2. Distillation - Train a smaller model

# Mistral
## Chat Template
Since one of the most common use case for LLMs is chat, rather than continuing a single string of text, 
the model instead continues a conversation.

Mistral uses a specific chat template called [ChatML](https://huggingface.co/docs/transformers/chat_templating).

Much like tokenization, different models expect very different input formats for chat. 
This is the reason for chat templates as a feature. 
Chat templates are part of the tokenizer. 
They specify how to convert conversations, represented as lists of messages, 
into a single tokenizable string in the format that the model expects.
