![Inspiring Image](https://repository-images.githubusercontent.com/629625919/db9030d2-92e5-4c14-a024-66b5af87e2fe)

# NexusLLM
# Introduction
NexusLLM is a GitHub repository dedicated to exploring various experiments related to Language Model Models (LMM). From fine-tuning and instruction-tuning to RAG and agent-based systems, it offers a diverse range of experiments and insights for researchers and enthusiasts interested in natural language processing and AI innovation.
# Setup
## Update PYTHONPATH
Add the current directory to the `PYTHONPATH` environment variables.
``` bash
export PYTHONPATH="$PYTHONPATH:/<absolute_path>/MediBioticsAI"
```

## Justfile
> `just` is a handy way to save and run project-specific commands
> 
> The main benefit it to keep all configuration and scripts in one place.
> 
> It uses the `.env` file for ingesting variables.

You can install it by following the [Documentation](https://just.systems/man/en/chapter_4.html).
Afterwards, you can execute existing commands located in the `justfile`.

Type `just` to list all available commands.


## Poetry

> Python packaging and dependency management made easy

### Installation

[Reference Documentation](https://python-poetry.org/)

Run the following command from the terminal:
``` bash
curl -sSL https://install.python-poetry.org | python3 -
```

For **MacOS** with ZSH add the `.local/bin` to the `PATH` environment variable. Modify the `.zshrc` file with the following command:

``` bash
export PATH="$HOME/.local/bin:$PATH"
```

### Add Dependency
``` bash
# NOTE: Use '--group dev' to install in the 'dev' dependencies list
poetry add <library_name>

poetry add <library> --group dev

poetry add <libarry> --group <group_name>
```

### Install Dependencies
``` bash
# Install the dependencies listed in pyproject.toml [tool.poetry.dependencies]
poetry install

# Use the option '--without test,docs,dev' if you want to esclude the specified group from install
poetry install --without test,docs,dev
```

# Resources
## ChatGPT
- [Andrej Karpathy Tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY)

## GPT-3

## HuggingFace Bloomz
- [Link to the resource](https://huggingface.co/bigscience/bloomz)

## LoRA
- [Link to a Tutorial from Phischmid](https://www.philschmid.de/fine-tune-flan-t5-peft)

