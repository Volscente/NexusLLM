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
- **Rank** - It is the number of Update Matrices (min = 8). It depends on the complexity of the dataset.
Higher the number, higher the complexity and, thus, the memory requirements.

## Advantages
1. Preserve pre-training weights, minimizing the risk of catastrophic forgetting
2. Update Matrices have far fewer parameters tha pre-training weights, thus are much more portable
3. Update Matrices are incorporated into original attention layers
4. Memory efficiency due to the dimension of Update Matrices