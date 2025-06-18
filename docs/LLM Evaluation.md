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

### Statistics
- They might perform poorly when the output implies reasoning capabilities

**List of Metrics:**
- **BLEU (BiLingual Evaluation Understudy)** - It evaluates the output of the LLM application against annotated ground truths. It calculates the precision for each matching n-gram (n consecutive words) between an LLM output and expected output to calculate their geometric mean and applies a brevity penalty if needed.
- **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** - It is used for text summarisation and calculates recall by comparing the overlap of n-grams between LLM outputs and expected outputs.