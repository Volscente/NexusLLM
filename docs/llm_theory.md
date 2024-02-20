# Terminology
## Auto-Regression
the way these models actually work is that after each token is produced, 
that token is added to the sequence of inputs. 

And that new sequence becomes the input to the model in its next step. 
This is an idea called *“Auto-Regression”*.

This feature is not always incorporated. For example, BERT does not have it.

# Fine-Tuning
## Definition
LLMs are *pre-trained* on very extensive text corpus
- LLaMa 2 on 2 trillion tokens
- BERT on BookCorpus (800M words) and Wikipedia (2500M words)

This pre-training is very long and costly.

Such pre-trained models are just able to predict tokens and, thus, construct sentences.
However, they're not really efficient in answering questions.

This is the reason for the *Fine-Tuning* step.

## Performance
A pure pre-trained model can be most of the time be out-performed by a fine-tuned model, even if the original pre-trainig
was performed on fewer tokens.

## Techniques
### Supervised Fine-Tuning (SFT)
Models are trained on a dataset of instructions and responses. 
It adjusts the weights in the LLM to minimize the difference between the generated answers and ground-truth responses, 
acting as labels.

It requires:
1. Good quality instruction dataset
2. Prompt template

### Reinforcement Learning from Human Feedback (RLHF)
Models learn by interacting with their environment and receiving feedback. 
They are trained to maximize a reward signal (using PPO), which is often derived from human evaluations 
of model outputs.

### Comparison
RLHF is able to better capture humans way of generating responses, but it's harder to implement.

# LoRA
## General
The *Low-Rang Adaptation* algorithm is designed for fine-tuning LLMs while keeping memory consumption low.

## Process
LLMs are pre-trained by updating the weights of certain weight matrices. 
LoRA focuses on pair of *Rank-Decomposition Weight Matrices* (Update matrices) 
to the existing pre-training weights. It basically adds new weights matrices on top.

### Hyperparameters
#### Rank
It is the number of Update Matrices (min = 8). It depends on the complexity of the dataset.
Higher the number, higher the complexity and, thus, the memory requirements.
To match a full fine-tune, the rank has to be equal to the model's hidden size (`model.config.hidden_size`)

#### Target Modules
It determines which weights and matrices have to be targeted. By default, the
*Query Vector* and *Value Vector*.
Such matrices can be usually retrieved as follow:
```python
from transformers import AutoModelForCausalLM
model_name = "huggyllama/llama-7b"      # can also be a local directory
model = AutoModelForCausalLM.from_pretrained(model_name)
layer_names = model.state_dict().keys()

for name in layer_names:
    print(name)

""" Output
model.embed_tokens.weight
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.0.mlp.gate_proj.weight
model.layers.0.mlp.down_proj.weight
model.layers.0.mlp.up_proj.weight
model.layers.0.input_layernorm.weight
model.layers.0.post_attention_layernorm.weight
...
model.norm.weight
lm_head.weight
"""
```

Naming convention is: `{identifier}.{layer}.{layer_number}.{component}.{module}.{parameter}`.

Some basic modules are:
- `up_proj`: The projection matrix used in the upward (decoder to encoder) attention pass. It projects the decoder's hidden states to the same dimension as the encoder's hidden states for compatibility during attention calculations.
down_proj: The projection matrix used in the downward (encoder to decoder) attention pass. It projects the encoder's hidden states to the dimension expected by thr decoder for attention calculations.
q_proj: The projection matrix applied to the query vectors in the attention mechanism. Transforms the input hidden states to the desired dimension for effective query representations.
v_proj: The projection matrix applied to the value vectors in the attention mechanism. Transforms the input hidden states to the desired dimension for effective value representations.
k_proj: The projection matrix applied to the key vectors blah blah.
o_proj: The projection matrix applied to the output of the attention mechanism. Transforms the combined attention output to the desired dimension before further processing.

## Advantages
1. Preserve pre-training weights, minimizing the risk of catastrophic forgetting
2. Update Matrices have far fewer parameters tha pre-training weights, thus are much more portable
3. Update Matrices are incorporated into original attention layers
4. Memory efficiency due to the dimension of Update Matrices