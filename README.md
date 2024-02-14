# Lawyer Assistant
LLMs based assistant that helps lawyers with their work.

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

- Install and register gitlab runner 
  - Set "Run untagged jobs" in the gitlab CI -> runners -> previously setup runner
  - Set concurrent = 30 in the /etc/gitlab-runner/config.toml

[Install elasticsearch with docker](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html)
If needed increase `vm.max_map_count` to at least 262144
Open `sudo nano /etc/sysctl.conf`, then add the line `vm.max_map_count=262144` at the end of file and then execute `sudo sysctl --system`

run elasticsearch: `docker run --name es01 --net elastic -v /home/jan/lawyer_assistant/elasticsearch_data_mountpoint:/usr/share/elasticsearch/data -p 9200:9200 -t -m 4GB docker.elastic.co/elasticsearch/elasticsearch:8.11.1`
run kibana: `docker run --name kib01 --net elastic -p 5601:5601 docker.elastic.co/kibana/kibana:8.11.1`

