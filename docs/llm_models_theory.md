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
