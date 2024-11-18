# Terminology
## Auto-Regression
the way these models actually work is that after each token is produced, 
that token is added to the sequence of inputs. 

And that new sequence becomes the input to the model in its next step. 
This is an idea called *“Auto-Regression”*.

This feature is not always incorporated. For example, BERT does not have it.

# Pre-Training
## Definition
It's the very first step of training a LLM and, in this operation, a huge amount of
text data is processed.

## Steps
### Data Processing
In order to feed the text data in the training process, they have to be converted into tokens
by a *Tokenizer*, which is specifically trained for the task.

Its job is to encode and decode text into tokens (and vice versa).

The dataset is then pre-processed using the tokenizer's vocabulary, 
converting the raw text into a format suitable for training the model. 
This step involves mapping tokens to their corresponding IDs, 
and incorporating any necessary special tokens or attention masks. 
Once the dataset is pre-processed, it is ready to be used for the pre-training phase.

### Training
In this step, the model learns either to predict the next token in a sequence,
or filling the missing tokens in a given sequence. In this way, the model learn  language patterns, grammar, and semantic relationships 

The task depends on the training algorithm, but it is a supervised-learning algorithm.

#### Learning Algorithms
- **Masked Language Modeling** - The model tries to predict certain masked tokens within the input sequence
- **Casual Language Modeling** - The model tries to predict the next token given the preceding context

# Fine-Tuning
## Definition
LLMs are *pre-trained* on very extensive text corpus
- LLaMa 2 on 2 trillion tokens
- BERT on BookCorpus (800M words) and Wikipedia (2500M words)

This pre-training is very long and costly.

Such pre-trained models are just able to predict tokens and, thus, construct sentences.
However, they're not really efficient in answering questions.

This is the reason for the *Fine-Tuning* step: allows us to specialize the model's capabilities and optimize its 
performance on a narrower, task-specific dataset.

## Process
The goal is to re-train the model's weights for a specific task. The way in which this happens
can vary much, depending on the Fine-Tuning algorithm chosen. The whole model's weights can be retrained, just a portion or
having another set of weights (LoRA).

During this Fine-Tuning Process, all the elements of a normal training are applied: optimizer (e.g., SGD or Adam), learning rate,
dropout, weight decay, overfit and early stopping.

## Dataset Definition
The dataset used for the Fine-Tuning should have:
1. **Data Diversity** - Do not address a single task, but aim for more. Ensure to include all possible conversation scenarios
2. **Dataset Size** - At least 10MiB. It's not easy to overfit a pre-trained model with fine-tuning, so the more, the better
3. **Dataset Quality** - Do not feed garbage

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

One example of dataset used in RLHF is [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf/viewer/default/train).
For each row there is one chosen and one rejected answer.

### Parameter Efficient Fine-Tuning (PEFT)
Since both SFT and RLHF are very costly, the PEFT was a huge step forward.
At a high-level, PEFT approaches append a significantly smaller set of weights (e.g., on the order of thousands of parameters) 
that are used to ‘perturb’ the pre-trained LLM weights. The perturbation has the effect of fine-tuning 
the LLM to perform a new task or set of tasks. This has the benefit of training a significantly smaller set of weights, 
compared to traditional fine-tuning of the entire model.

Some PEFT techniques are:
- **Adapter-based fine-tuning** - It employs small modules, called adapters, to the pre-trained model. Only adapters' parameters are trained
- **Low-Rank Adaptation (LoRA)** - It uses two smaller matrices to approximate the original weight matrix update instead 
of fine-tuning the whole LLM. This technique freezes the original weights and trains these update matrices, 
significantly reducing resource requirements with minimum additional inference latency. 
Additionally, LoRA has improved variants such as QLoRA,48 which uses quantized weights for even greater efficiency. 

### Comparison
RLHF is able to better capture humans way of generating responses, but it's harder to implement.

# LoRA
## General
The *Low-Rang Adaptation* algorithm is designed for fine-tuning LLMs while keeping memory consumption low.

The main concept is the *Low Rank*: there are very few elements in the weights matrices of an LLM that carry information.
So it is required to just add a Low Rank Update matrices in order to capture such valuable information.

The main point to fine-tune is the Self-Attention layer, since it's the one having the lowest rank (i.e, the most redundant information)

## Process
LLMs are pre-trained by updating the weights of certain weight matrices, added into the model's architecture. 
LoRA focuses on pair of *Rank-Decomposition Weight Matrices* (Update matrices) 
to the existing pre-training weights. It basically adds new weights matrices on top.

The update matrices are placed only for the self-attention layers.
It also shows better results when these Update Matrices are applied on the Value Tensors of the self-attention layers.

If the rank is equal to the rank of the Self-Attention layer, we are doing a full fine-tuning.

### Hyperparameters
#### Rank
It is the number of independent rows within a matrix. It depends on the complexity of the dataset.
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
model.layers.0.self_attn.rotary_emb.inv_freq
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
- `up_proj`: The projection matrix used in the upward (decoder to encoder) attention pass. 
It projects the decoder's hidden states to the same dimension as the encoder's hidden states 
for compatibility during attention calculations.
- `down_proj`: The projection matrix used in the downward (encoder to decoder) attention pass. 
It projects the encoder's hidden states to the dimension expected by thr decoder for attention
calculations.
- `q_proj`: The projection matrix applied to the query vectors in the attention mechanism.
Transforms the input hidden states to the desired dimension for effective query representations.
- `v_proj`: The projection matrix applied to the value vectors in the attention mechanism. 
Transforms the input hidden states to the desired dimension for effective value representations.
- `k_proj`: The projection matrix applied to the key vectors blah blah.
Transforms the input hidden states to the desired dimension for effective value representations.
- `o_proj`: The projection matrix applied to the output of the attention mechanism. 
Transforms the combined attention output to the desired dimension before further processing.

## Advantages
1. Preserve pre-training weights, minimizing the risk of catastrophic forgetting
2. Update Matrices have far fewer parameters tha pre-training weights, thus are much more portable
3. Update Matrices are incorporated into original attention layers
4. Memory efficiency due to the dimension of Update Matrices


# QLoRA
## Definition
QLoRA (Quantized Low Rank Adapters) is an efficient fine-tuning approach that reduces memory usage 
while maintaining high performance for large language models. 
It enables the fine-tuning of a 65B parameter model on a single 48GB GPU, 
while preserving full 16-bit fine-tuning task performance.

**Key Innovations:**
- Backpropagation of gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters (LoRA)
- Use of a new data type called 4-bit NormalFloat (NF4), which optimally handles normally distributed weights
- Double quantization to reduce the average memory footprint by quantising the quantization constants
- Paged optimizers to effectively manage memory spikes during the fine-tuning process

## Hyperparameters
- **Batch Size** - Number of training sample processed at the same time, before updating the weights
- **Epochs** - The number of time the model sees the entire dataset

# Axolotl
## Definition
Axolotl is a tool designed to streamline the fine-tuning of various AI models, 
offering support for multiple configurations and architectures. It supports different moodels (e.g., LLaMA, Falcon, etc.)
and different Fine-Tuning algorithms, such as LoRA and QLoRA.

## Advantages
- **Single Configuration File** - All parameters used to train an LLM are neatly stored in a yaml config file. 
This makes it convenient for sharing and reproducing models.
Examples in the GitHub repo [here](https://github.com/OpenAccess-AI-Collective/axolotl/tree/main/examples).
- **Dataset Flexibility** - Different prompt's formats supported

## Modify Configuration
Some parameters that should be usually changed are:
- `base_model`
- `base_model_config` (Usually the same as the `base_model`)
- `hub_model_id` (New model name)
- `datasets` (`path` and `type`)

### Gradient Checkpointing
Gradient checkpointing is a technique used in machine learning, particularly in deep learning, 
to reduce the memory requirements during the training of neural networks, especially those with many layers.

In deep learning, during the backpropagation process, 
gradients are computed and stored for each layer in order to update the model parameters. 
However, as neural networks become deeper, 
the memory requirements for storing these gradients can become a limiting factor, 
especially in memory-constrained environments such as GPUs.

Gradient checkpointing addresses this issue by trading off memory consumption with recomputation during the backward pass. 
Instead of storing the gradients for all layers, 
only a subset of the layers' activations and intermediate gradients are stored, 
while the remaining layers' activations are recomputed during the backward pass when needed. 

# Direct Preference Optimisation (DPO)
## Definition
It is RLHF Fine-Tuning technique.

## PPO vs DPO
The *Proximal Policy Optimization (PPO)* is another reinforcement learning algorithm that aims to improve the policy of an agent.
One of the key features of PPO is the use of a "proximal policy optimization" approach, 
which constrains the policy updates to be within a certain "trust region" 
to prevent large policy changes that could lead to instability in learning.
It however still very unstable and computationally expensive.

It also implies optimizing a surrogate objective function that approximates the true objective.

The *Direct Preference Optimization (DPO)* directly optimizes the preferences of an agent over different actions or policies, 
rather than optimizing a surrogate objective function.
In DPO, the agent learns a preference function that assigns values or scores to different actions or 
policies based on their expected long-term rewards.

# Code Examples
## SFT on Vertex AI
```python
Python
  # Before you start run this command:
  # pip install --upgrade --user --quiet google-cloud-aiplatform
  # after running pip install make sure you restart your kernel
  import vertexai
  from vertexai.generative_models import GenerativeModel
  from vertexai.preview.tuning import sft
  # TODO : Set values as per your requirements
  # Project and Storage Constants
  PROJECT_ID = ‘<project_id>’
  REGION = ‘<region>’
  vertexai.init(project=PROJECT_ID, location=REGION)
  # define training & eval dataset.
  TRAINING_DATASET = ‘gs://cloud-samples-data/vertex-ai/model-evaluation/
  peft_train_sample.jsonl’
  # set base model and specify a name for the tuned model
  BASE_MODEL = ‘gemini-1.5-pro-002’
  TUNED_MODEL_DISPLAY_NAME = ‘gemini-fine-tuning-v1’
  # start the fine-tuning job
  sft_tuning_job = sft.train(
     source_model=BASE_MODEL,
     train_dataset=TRAINING_DATASET,
     # # Optional:
     tuned_model_display_name=TUNED_MODEL_DISPLAY_NAME,
)
  # Get the tuning job info.
  sft_tuning_job.to_dict()
  # tuned model endpoint name
  tuned_model_endpoint_name = sft_tuning_job.tuned_model_endpoint_name
  # use the tuned model
  tuned_genai_model = GenerativeModel(tuned_model_endpoint_name)
  print(tuned_genai_model.generate_content(contents=’What is a LLM?’))
```