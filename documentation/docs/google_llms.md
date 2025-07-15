# Services
## AI Studio
AI Studio is free to use and only requires a compatible Google account to log in and get started. It is deeply 
integrated with the Gemini API, which comes with a generous free tier that you can use to run.

# Python SDK
## Installation
```bash
pip install -U -q "google-generativeai>=0.8.3"
```
```pyhont
# Import
import google.generativeai as genai
```

## Setup
```python
# Import Standard Libraries
import google.generativeai as genai

# Set the API Key from AI Studio
genai.configure(api_key=GOOGLE_API_KEY)
```