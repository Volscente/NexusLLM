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
