# Hugging Face
## Transformers Library
### Pipeline
The `pipeline()` function allows  us to use Transformer models.

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")
```
There are various flavours of such a function.

**Text Pipelines**
- `text-generation`: Generate text from a prompt
- `text-classification`: Classify text into predefined categories
- `summarization`: Create a shorter version of a text while preserving key information
- `translation`: Translate text from one language to another
- `zero-shot-classification`: Classify text without prior training on specific labels
- `feature-extraction`: Extract vector representations of text
- `ner`: Find which parts of the input text correspond to entities
- `question-answering`: Answer questions from a given context`

**Image pipelines**
- `image-to-text`: Generate text descriptions of images
- `image-classification`: Identify objects in an image
- object-detection`: Locate and identify objects in images

**Audio pipelines**
- `automatic-speech-recognition`: Convert speech to text
- `audio-classification`: Classify audio into categories
- `text-to-speech`: Convert text to spoken audio

**Multimodal pipelines**
- `image-text-to-text`: Respond to an image based on a text prompt