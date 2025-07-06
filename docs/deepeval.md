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

## LLM Evaluation
### General
It is composed by:
- Test Cases
```python
from deepeval.test_case import LLMTestCase

test_case = LLMTestCase(
  input="Who is the current president of the United States of America?",
  actual_output="Joe Biden",
  retrieval_context=["Joe Biden serves as the current president of America."]
)
```
- Metrics
```python
from deepeval.metrics import AnswerRelevancyMetric

answer_relevancy_metric = AnswerRelevancyMetric()
```
- Evaluation Datasets (Check the dedicated section)

Running a "Test Run":
```python
answer_relevancy_metric.measure(test_case)
print(answer_relevancy_metric.score)
```

### Types
- End-to-end evaluation
- Component-level evaluation

# Metrics
## General
There are two types of metrics:
- Out-of-the-Box stored in `deepeval.metrics`
- Custom metrics &rarr; defined by `deepeval.metrics.GEval` (Non-deterministic) or `deepeval.metrics.DAGMetric` (Deterministic)

# Datasets
## General
- They are evaluation datasets instance from `EvaluationDataset` (groups together multiple test cases of a same category)
- Either `LLMTestCase` or `Goldens` (no `actual_output`) instances

## Types
### Goldens
- Allow for LLM output generation during evaluation time &rarr; That's why they don't have `actual_output`
- Serve as templates before becoming fully-formed test cases

## TestCases
### Single-Turn
It tests a single, atomic unit of interaction, either between LLM's components or users.

It can either implement an End-to-end evaluation or a Component-level evaluation.

```python
from deepeval.test_case import LLMTestCase, ToolCall

test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    expected_output="You're eligible for a 30 day refund at no extra cost.",
    actual_output="We offer a 30-day full refund at no extra cost.",
    context=["All customers are eligible for a 30 day full refund at no extra cost."],
    retrieval_context=["Only shoes can be refunded."], # Retrieved documents in a RAG
    tools_called=[ToolCall(name="WebSearch")]
)
```

Other useful parameters are:
- `token_cost`
- `completion_time`

### Tools
The `tools_called` is a list of `ToolCall` objects, which are Pydantic types:
```python
class ToolCall(BaseModel):
    name: str
    description: Optional[str] = None
    reasoning: Optional[str] = None # How to use the tool
    output: Optional[Any] = None # Tool's output - Any data type
    input_parameters: Optional[Dict[str, Any]] = None
```

An example:
```python
tools_called=[
    ToolCall(
        name="Calculator Tool"
        description="A tool that calculates mathematical equations or expressions.",
        input={"user_input": "2+3"}
        output=5
    ),
    ToolCall(
        name="WebSearch Tool"
        reasoning="Knowledge base does not detail why the chicken crossed the road."
        input={"search_query": "Why did the chicken crossed the road?"}
        output="Because it wanted to, duh."
    )
]
```

### MLLM Test Case
An MLLMTestCase in deepeval is designed to unit test outputs from MLLM (Multimodal Large Language Model) applications.

Example:
```python
from deepeval.test_case import MLLMTestCase, MLLMImage

mllm_test_case = MLLMTestCase(
    # Replace this with your user input
    input=["Change the color of the shoes to blue.", MLLMImage(url="./shoes.png", local=True)]
    # Replace this with your actual MLLM application
    actual_output=["The original image of red shoes now shows the shoes in blue.", MLLMImage(url="https://shoe-images.com/edited-shoes", local=False)]
)
```

### Multi-Turn
A multi-turn test case in deepeval is represented by a `ConversationalTestCase`, and has TWO parameters:
- `turns`
- `chatbot_role`

```python
# Turn class definition
class Turn:
    role: Literal["user", "assistant"]
    content: str
    user_id: Optional[str] = None
    retrieval_context: Optional[List[str]] = None
    tools_called: Optional[List[ToolCall]] = None
    additional_metadata: Optional[Dict] = None

# Example
from deepeval.test_case import Turn, ConversationalTestCase

turns = [
    Turn(
        role="assistant",
        content="Why did the chicken cross the road?",
    ),
    Turn(
        role="user",
        content="Are you trying to be funny?",
    ),
]

test_case = ConversationalTestCase(turns=turns)
```

## Usage
### Creation
```python
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset, Golden

# Dataset creation from LLMTestCases
first_test_case = LLMTestCase(input="...", actual_output="...")
second_test_case = LLMTestCase(input="...", actual_output="...")

dataset = EvaluationDataset(test_cases=[first_test_case, second_test_case])

# Dataset creation from Goldens
first_golden = Golden(input="...")
second_golden = Golden(input="...")

dataset_goldens = EvaluationDataset(goldens=[first_golden, second_golden])
print(dataset_goldens.goldens)

# Append
dataset.test_cases.append(test_case)
# or
dataset.add_test_case(test_case)
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

### Loading
```python
# From JSON
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()

# Add as test cases
dataset.add_test_cases_from_json_file(
    # file_path is the absolute path to you .json file
    file_path="example.json",
    input_key_name="query",
    actual_output_key_name="actual_output",
    expected_output_key_name="expected_output",
    context_key_name="context",
    retrieval_context_key_name="retrieval_context",
)

# Or, add as goldens
dataset.add_goldens_from_json_file(
    # file_path is the absolute path to you .json file
    file_path="example.json",
    input_key_name="query"
)

# From CSV
# Add as test cases
dataset.add_test_cases_from_csv_file(
    # file_path is the absolute path to you .csv file
    file_path="example.csv",
    input_col_name="query",
    actual_output_col_name="actual_output",
    expected_output_col_name="expected_output",
    context_col_name="context",
    context_col_delimiter= ";",
    retrieval_context_col_name="retrieval_context",
    retrieval_context_col_delimiter= ";"
)

# Or, add as goldens
dataset.add_goldens_from_csv_file(
    # file_path is the absolute path to you .csv file
    file_path="example.csv",
    input_col_name="query"
)
```

### PyTest Integration
```python
import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric



# Loop through test cases using Pytest
@pytest.mark.parametrize(
    "test_case",
    dataset,
)
def test_customer_chatbot(test_case: LLMTestCase):
    hallucination_metric = HallucinationMetric(threshold=0.3)
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    assert_test(test_case, [hallucination_metric, answer_relevancy_metric])


@deepeval.on_test_run_end
def function_to_be_called_after_test_run():
    print("Test finished!")
```

### End-to-End
```python
# Define you LLM application
def your_llm_app(input: str):
    print("Call LLM!")

# Define the Dataset for evaluation
goldens = [Golden(input="...")]

# Create the test cases
test_case = []
for golden in goldens:
    res, text_chunks = your_llm_app(golden.input) # Call the LLM to generate the output and maybe a RAG context
    test_case = LLMTestCase(input=golden.input, actual_output=res, retrieval_context=text_chunks)

# Evaluate end-to-end
evaluate(test_cases=test_cases, metrics=[AnswerRelevancyMetric()])
```

### Save
```python
# Locally
dataset.save_as(file_type="csv", directory="./deepeval-test-dataset", include_test_cases=True)
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