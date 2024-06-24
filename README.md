# RL Agent with the world-and-self model
Reinforcement learning agent with the world model that includes agent's internal state, current time and randomness source.
World-and-self model is based on the transformer architecture.

### Project setup

You may need to install python helper libraries before installing the python:
- `sudo apt install -y python3.11-dev`
- `sudo apt install -y python3.11-tk`
- `sudo apt install -y liblzma-dev`


- Install [pyenv](https://github.com/pyenv/pyenv)
- Install Python 3.11.5 (via pyenv):
  - You may need to run `sudo apt install -y python3.11-dev python3.11-tk liblzma-dev libsqlite3-dev` before
- Set python version: `pyenv local 3.11.5`
- Install [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)

- Install project dependencies: `poetry install`
- Install [pre-commit](https://pre-commit.com/#install) on the system
- Install pre-commit for the project with: `pre-commit install` in the project root dir

- Install [PyCharm](https://www.jetbrains.com/pycharm/)
- This project uses Google style docstrings, set it in the Pycharm settings | Tools | Python Integrated Tools and also check following checkboxes:
  - Analyze Python code in docstrings
  - Render external documentation for stdlib
