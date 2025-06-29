# Introduction
## Installation
```
# Install
pip install deepeval

# Cloud login (e.g., Confident AI)
```

- By default, DeepEval uses OpenAI models for evaluation.
- Set the OpenAI key in `OPENAI_API_KEY`.
- It is possible to customise the back-end LLM.

Set the output folder:
```
# linux
export DEEPEVAL_RESULTS_FOLDER="./data"

# or windows
set DEEPEVAL_RESULTS_FOLDER=.\data
```

## Commands
```
# Run test
deepeval test run test_example.py
```

# Code Snippets
## Basic Usage
```python
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

def test_correctness():
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5
    )
    test_case = LLMTestCase(
        input="I have a persistent cough and fever. Should I be worried?",
        # Replace this with the actual output from your LLM application
        actual_output="A persistent cough and fever could be a viral infection or something more serious. See a doctor if symptoms worsen or don't improve in a few days.",
        expected_output="A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical attention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty breathing, chest pain, or other concerning signs."
    )
    assert_test(test_case, [correctness_metric])
```