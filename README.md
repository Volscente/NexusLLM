![Inspiring Image](https://repository-images.githubusercontent.com/629625919/db9030d2-92e5-4c14-a024-66b5af87e2fe)

# NexusLLM
# Introduction
NexusLLM is a GitHub repository dedicated to exploring various experiments related to Language Model Models (LMM). From fine-tuning and instruction-tuning to RAG and agent-based systems, it offers a diverse range of experiments and insights for researchers and enthusiasts interested in natural language processing and AI innovation.

# Resources
The full documentation of the project can be found in the dedicated [GitHub Pages]().

For the developers, check the wiki [Package & Modules](https://github.com/Volscente/NexusLLM/wiki/Package-&-Modules) Section.

Please refer to this [Contributing Guidelines](https://github.com/Volscente/NexusLLM/wiki/Contributing-Guidelines) in order to contribute to the repository.

# Setup
## Environment Variables
Add the project root directory as `NEXUSLLM_ROOT_PATH` environment variable.
``` bash
export NEXUSLLM_ROOT_PATH="/<absolute_path>/NexusLLM"
```
Create a `.env` file in the root folder like
```
# Set the Root Path
NEXUSLLM_ROOT_PATH="/<absolute_path>/NexusLLM"
```

## Setup gcloud CLI
Install `gcloud` on the local machine ([Guide](https://cloud.google.com/sdk/docs/install)).

Authenticate locally to GCP:
```bash
gcloud auth login
```

Set the project ID.
```bash
# List all the projects
gcloud projects list

# Set the project
gcloud config set project <project_id>
```

Create authentication keys.
```bash
gcloud auth application-default login
```

## Justfile
> `just` is a handy way to save and run project-specific commands
> 
> The main benefit it to keep all configuration and scripts in one place.
> 
> It uses the `.env` file for ingesting variables.

You can install it by following the [Documentation](https://just.systems/man/en/chapter_4.html).
Afterward, you can execute existing commands located in the `justfile`.

Type `just` to list all available commands.

## Pre-commit
```bash
# Install
pre-commit install

# Check locally
pre-commit run --all-files
```