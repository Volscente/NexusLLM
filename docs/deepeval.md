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

## Additional Libraries
- The `deepteam` includes any security related testing

# Metrics
## General
There are two types of metrics:
- Out-of-the-Box stored in `deepeval.metrics`
- Custom metrics &rarr; defined by `deepeval.metrics.GEval` (Non-deterministic) or `deepeval.metrics.DAGMetric` (Deterministic)

# Datasets
## General
- They are evaluation datasets instance from `EvaluationDataset`
- Either `LLMTestCases` or `Goldens` (no `actual_output`) instances

## Usage
### Creation
```python
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset(test_cases=[LLMTestCase(input="...", actual_output="...")])
```

- Pull it from the Cloud
```python
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric

dataset = EvaluationDataset()
# supply your dataset alias
dataset.pull(alias="QA Dataset")

evaluate(dataset, metrics=[AnswerRelevancyMetric()])
```

- Generate synthetic data
```python
from deepeval.synthesizer import Synthesizer
from deepeval.dataset import EvaluationDataset

synthesizer = Synthesizer()
goldens = synthesizer.generate_goldens_from_docs(
  document_paths=['example.txt', 'example.docx', 'example.pdf']
)

dataset = EvaluationDataset(goldens=goldens)
```

### Evaluate Function
- Evaluate over the entire dataset
```python
from deepeval.metrics import AnswerRelevancyMetric
from deepeval import evaluate
...

evaluate(dataset, [AnswerRelevancyMetric()])
```

- Run the evaluation in parallel
```bash
deepeval test run test_dataset.py -n 2
```


### PyTest Integration
```python
import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
...

# Loop through test cases using Pytest
@pytest.mark.parametrize(
    "test_case",
    dataset,
)
def test_customer_chatbot(test_case: LLMTestCase):
    assert_test(test_case, [AnswerRelevancyMetric(threshold=0.5)])
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
    assert_test(test_case, [correctness_metric]) # Possible to specify multiple metrics
```

## Observe Decorator
It is used to evaluate and keep track of the evaluation of single app's components

```python
from deepeval.tracing import observe, update_current_span
from deepeval.test_case import LLMTestCase
from deepeval.dataset import Golden
from deepeval.metrics import GEval
from deepeval import evaluate

correctness = GEval(name="Correctness", criteria="Determine if the 'actual output' is correct based on the 'expected output'.", evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT])

@observe(metrics=[correctness])
def inner_component():
    # Component can be anything from an LLM call, retrieval, agent, tool use, etc.
    update_current_span(test_case=LLMTestCase(input="...", actual_output="..."))
    return

@observe
def llm_app(input: str):
    inner_component()
    return

evaluate(observed_callback=llm_app, goldens=[Golden(input="Hi!")])
```