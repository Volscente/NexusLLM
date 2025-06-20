# General
## Areas of Evaluation
An evaluation framework for LLMs should target two main areas:
- **Use Case** - Custom metrics that directly measure how well the LLM is performing regarding the specific task
- **System Architecture** - Generic metrics on, for example, faithfulness of information retrieved by the RAG or the Task Completion for AI Agents

# Metrics
## List
- **Answer Relevancy** - Determines whether an LLM output is able to address the given input and certain context and rules ✅
- **Task Completion** - Determines whether an LLM agent is able to complete the task it was set out to do ⚠️ &rarr; How to determine completion state?
- **Correctness** - Determines whether an LLM output is factually correct based on some ground truth ✅
- **Hallucination** - Determines whether an LLM output contains fake or made-up information ❌ &rarr; Impossible to determine
- **Tool Correctness** - Determines whether an LLM agent is able to call the correct tools for a given task ⚠️ &rarr; How to determine if the tool is correct?
- **Contextual Relevancy** - Determines whether the retriever in a RAG-based LLM system is able to extract the most relevant information for your LLM as context ✅ &rarr; Similar to the first one, but on each retrieved document
- **Responsible Metrics** - Includes metrics such as bias and toxicity, which determines whether an LLM output contains (generally) harmful and offensive content ✅
- **Task-Specific Metrics** - Includes metrics such as summarization, which usually contains a custom criteria depending on the use-case ✅

## Types
Some metrics are based on Statistics, while others are sometimes referred as *"Model-based"*:

![LLM Metrics](./images/llm_metrics.png)

### Statistics Metrics
- They might perform poorly when the output implies reasoning capabilities (No semantic is included)
- They do not take into account any

**List of Metrics:**
- **BLEU (BiLingual Evaluation Understudy)** - It evaluates the output of the LLM application against annotated ground truths. It calculates the precision for each matching n-gram (n consecutive words) between an LLM output and expected output to calculate their geometric mean and applies a brevity penalty if needed.
- **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** - It is used for text summarisation and calculates recall by comparing the overlap of n-grams between LLM outputs and expected outputs.  It also leverages external linguistic databases like WordNet to account for synonyms. The final score is the harmonic mean of precision and recall, with a penalty for ordering discrepancies.
- **METEOR (Metric for Evaluation of Translation with Explicit Ordering)** - It calculates scores by assessing both precision (n-gram matches) and recall (n-gram overlaps), adjusted for word order differences between LLM outputs and expected outputs. It can also leverages exteral linguistic databases.
- **Levenshtein distance**

### Model-based Metrics
- Reliable but inaccurate (struggle to keep semantic included), because of their probabilistic nature

**List of Metrics:**
- **NLI** - It is a Non-LLM based and uses Natural Language Inference models to classify whether an LLM output is logically consistent (entailment), contradictory, or unrelated (neutral) with respect to a given reference text.
- **BLEURT (Bilingual Evaluation Understudy with Representations from Transformers)** - It uses pre-trained models like BERT to score LLM outputs on some expected outputs

### Statistical and Model-Based Scorers
- **BERTScore** - It relies on a pre-trained LLM like BERT and on the cosine similarity between expected output and predicted output. Afterward, the similarities are aggregated to produce a final score.
- **MoverScore** - It relies on LLM like BERT to obtain deeper contextualised word embeddings for both reference text and generated text before computing the similarity.

# G-Eval (Model-based Scorer)
## Introduction
- It is an LLM-based Scorer ([Paper](https://arxiv.org/pdf/2303.16634.pdf))
- Documentation from [DeepEval](https://www.deepeval.com/docs/metrics-llm-evals)

![G-Eval](./images/g_eval.png)

## Process
1. Prompt with the following information: 1) Task Introduction; 2) Evaluation Criteria
2. Generate through the previous output the list of Evaluation Steps through the *"Auto Chain of Thoughts"*
3. Prompt the Scorer LLM with
   - Evaluation Steps
   - Input Context
   - Input Target
4. (Optional) Normalise the output score by the probabilities of the output tokens

## Code Snippets
### Basic Implementation
```python
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

test_case = LLMTestCase(input="input to your LLM", actual_output="your LLM output")
coherence_metric = GEval(
    name="Coherence",
    criteria="Coherence - the collective quality of all sentences in the actual output",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
)

coherence_metric.measure(test_case)
print(coherence_metric.score)
print(coherence_metric.reason)
```

# DAG (Model-based Scorer)
## Introduction
- Deep Acyclic Graph is a LLM-based scorer that relies on a decision tree
- Each node is an LLM Judgement and each edge is a decision
- Each leaf node is associated with a hardcoded score

![DAG LLM Scorer](./images/dag_llm.png)

## Advantages
- Slightly more deterministic, since there's a certain degree of control in the score determination
- It can be used to filter away edge cases where LLM output doesn't even meet minimum requirements

## Code Snippets
### Basic Implementation
```python
from deepeval.test_case import LLMTestCase
from deepeval.metrics.dag import (
    DeepAcyclicGraph,
    TaskNode,
    BinaryJudgementNode,
    NonBinaryJudgementNode,
    VerdictNode,
)
from deepeval.metrics import DAGMetric

correct_order_node = NonBinaryJudgementNode(
    criteria="Are the summary headings in the correct order: 'intro' => 'body' => 'conclusion'?",
    children=[
        VerdictNode(verdict="Yes", score=10),
        VerdictNode(verdict="Two are out of order", score=4),
        VerdictNode(verdict="All out of order", score=2),
    ],
)

correct_headings_node = BinaryJudgementNode(
    criteria="Does the summary headings contain all three: 'intro', 'body', and 'conclusion'?",
    children=[
        VerdictNode(verdict=False, score=0),
        VerdictNode(verdict=True, child=correct_order_node),
    ],
)

extract_headings_node = TaskNode(
    instructions="Extract all headings in `actual_output`",
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    output_label="Summary headings",
    children=[correct_headings_node, correct_order_node],
)

# create the DAG
dag = DeepAcyclicGraph(root_nodes=[extract_headings_node])

# create the metric
format_correctness = DAGMetric(name="Format Correctness", dag=dag)

# create a test case
test_case = LLMTestCase(input="your-original-text", actual_output="your-summary")

# evaluate
format_correctness.measure(test_case)
print(format_correctness.score, format_correctness.reason)
```

# Prometheus (Model-based Scorer)
## Introduction
- LLM-based evaluation framework use case agnostic
- Based con Llama-2-chat and fine-tuned for evaluation purposes

## Advantages
- Evaluation steps are not produced by LLM, but are embedded in the node itself

# QAG Score (Hybrid Scorer)
## Introduction
QAG (Question Answer Generation) Score uses binary answer (‘yes’ or ‘no’) to close-ended questions (which can be generated or preset) to compute a final metric score.

**Example:**
```
So for this example LLM output:

Martin Luther King Jr., the renowned civil rights leader, was assassinated on April 4, 1968, at the Lorraine Motel in Memphis, Tennessee. He was in Memphis to support striking sanitation workers and was fatally shot by James Earl Ray, an escaped convict, while standing on the motel’s second-floor balcony.
A claim would be:

Martin Luther King Jr. assassinated on the April 4, 1968
And a corresponding close-ended question would be:

Was Martin Luther King Jr. assassinated on the April 4, 1968?
```
## Advantages
- The score is not directly generated by an LLM

# GPTScore
## Introduction
It is similar to G-Eval, but the evaluation trask is performed with a form-filling paradigm.

![GPTScore](./images/gptscore.png)

# SelfCheckGPT
## Introduction
It samples multiple output in order to detect hallucinations through a model-based approach.

![SelfCheckGPT](./images/selfcheckgpt.png)

