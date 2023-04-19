# Large Language Models Experimentation
# Introduction
A repository containing research and experimentations with LLMs

# Setup
## Poetry
### Definition
Poetry is a tool for dependency management and packaging in Python. 
It allows you to declare the libraries your project depends on and it will manage (install/update) them for you. Poetry offers a lockfile to ensure repeatable installs, and can build your project for distribution.

[Reference Documentation](https://python-poetry.org/)

### Installation
#### MacOS
Run the following command from the terminal:
``` bash
curl -sSL https://install.python-poetry.org | python3 -
```

For **MacOS** with ZSH add the `.local/bin` to the `PATH` environment variable. Modify the `.zshrc` file with the following command:

``` bash
export PATH="$HOME/.local/bin:$PATH"
```
### Pyproject.toml

``` toml
[tool.poetry]
name = "rp-poetry"
version = "0.1.0"
description = ""
authors = ["Philipp <philipp@realpython.com>"]

[tool.poetry.dependencies]
python = "^3.9"

[tool.poetry.dev-dependencies]
pytest = "^5.2"

[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"
```

Each section identified by square brackets is called "Table". If a table is too-specific, it must be prefixed with tool. 
You now see that the only tool is poetry, but you might also have `[tool.pytest.ini_options]` for pytest.

### Poetry.lock
If you have never run the command before and there is also no `poetry.lock` file present, 
Poetry simply resolves all dependencies listed in your `pyproject.toml` file and downloads the latest version of their files.

When Poetry has finished installing, it writes all the packages and their exact versions that it downloaded to the `poetry.lock` file, 
locking the project to those specific versions. 
You should commit the `poetry.lock` file to your project repo so that all people working on the project are locked to
the same versions of dependencies (more below).

### Create Virtual Environment
Poetry is able to manage virtual environments and it is not created by default when creating a new poetry project. 

**NOTE:** PyCharm will ask you if you want to create one.

Otherwise, inside the project directory, use the following command:
``` bash
poetry env use python3
```

The virtual environment will be created in the directory `~/Library/Caches/pypoetry` for MacOS.

The created venvs can be viewed with the following command:
``` bash
poetry env list
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
## GPT-3

## HuggingFace Bloomz
- [Link to the resource](https://huggingface.co/bigscience/bloomz)

## LoRA
- [Link to a Tutorial from Phischmid](https://www.philschmid.de/fine-tune-flan-t5-peft)

