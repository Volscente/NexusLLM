# Introduction
## Definition
Agents are Generative AI models that can be trained to use tools
to access real-time information or suggest a real-world action.

This combination of reasoning, logic, and access to external information
that are all connected to a Generative AI model invokes the concept of an agent, or a
program that extends beyond the standalone capabilities of a Generative AI model.

A Generative AI agent can be defined as an application that
attempts to achieve a goal by observing the world and acting upon it using the tools that it
has at its disposal.

![Agent Architecture](./images/agent_architecture.png)

## Agents vs. Models

![Agents vs. Models](./images/agents_vs_models.png)

# Architecture
## Structure
An agent is composed by three main components:
- The Model - It's a LLM that acts as a centralised decision maker, thanks to techniques as ReAct, Chain-of-Thought or Tree-of-Thought
- The Tools - Foundational models remain constrained by their inability to interact with the outside world. Tools bridge this gap (e.g., RAG).
- The Orchestration Layer - The orchestration layer describes a cyclical process that governs how the agent takes in
information, performs some internal reasoning, and uses that reasoning to inform its next
action or decision. It performs this loop until the end goal is reached.

## Process
The sequence of events might go something like this:
1. User sends query to the agent
2. Agent begins the ReAct sequence
3. The agent provides a prompt to the model, asking it to generate one of the next ReAct
steps and its corresponding output:

   a. **Question**: The input question from the user query, provided with the prompt 

   b. **Thought**: The model’s thoughts about what it should do next c. Action: The model’s decision on what action to take next

   c. **Action**: The model's decision on what action to take next

       i. This is where tool choice can occur

       ii. For example, an action could be one of [Flights, Search, Code, None], where the first
       3 represent a known tool that the model can choose, and the last represents “no
       tool choice”

   d. **Action Input**: The model's decision on what inputs to provide to the tool (if any)

   e. **Observation**: The result of the action / action input sequence

       i. This though / action / action input / observation could repeat N-times as needed

   f. **Final Answer**: The model's final answer to provide to the original user query
4. The ReAct loop concludes and a final answer is provided back ot the user