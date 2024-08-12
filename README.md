# BERT Score API

## Project Structure

- It uses `project.toml` instead of `setup.py` and `setup.cfg`. The reasoning is following:
    - As [official setuptools guide](https://github.com/pypa/setuptools/blob/main/docs/userguide/quickstart.rst) says, " configuring new projects via setup.py is discouraged"
    - One of the biggest problems with setuptools is that the use of an executable file (i.e. the setup.py) cannot be executed without knowing its dependencies. And there is really no way to know what these dependencies are unless you actually execute the file that contains the information related to package dependencies.
    - The pyproject.toml file is supposed to solve the build-tool dependency chicken and egg problem since pip itself can read pyproject.yoml along with the version of setuptools or wheel the project requires.
    - The pyproject.toml file was introduced in PEP-518 (2016) as a way of separating configuration of the build system from a specific, optional library (setuptools) and also enabling setuptools to install itself without already being installed. Subsequently PEP-621 (2020) introduces the idea that the pyproject.toml file be used for wider project configuration and PEP-660 (2021) proposes finally doing away with the need for setup.py for editable installation using pip.
- It uses [rye](https://github.com/astral-sh/rye) for python dependency operations and virtual environment management.
- It uses `src` layout, which is the recommended layout for python projects to avoid common [pitfalls](https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure).

## Development Installation

- Install rye - System wide

```bash
make -s install-rye
```

- Install the project dependencies

```bash
make -s install
```

## Test the API

```bash

curl -X POST "http://127.0.0.1:8888/score_calculation/all" \
-H "Content-Type: application/json" \
-d '{
    "candidate": ["The quick brown fox jumps over the lazy dog"],
    "reference": ["A fast brown fox leaps over a lazy dog"]
}'

```

- You can also find example `Client` implementation [here](https://github.com/ilkersigirci/bert-score-api/blob/main/src/bert_score_api/client.py)


## Production Usage with Docker

- One can use below `docker-compose` file.
- `docker compose pull bert-score-api` to pull the image.
- `docker compose up bert-score-api` to run the service.
- To run command inside the container:

```bash
docker run -it ghcr.io/ilkersigirci/bert-score-api:latest bash

# Temporary container
docker run --rm -it ghcr.io/ilkersigirci/bert-score-api:latest bash
```

```yaml
networks:
  bert-score-api-net:
    name: bert-score-api-net
    driver: bridge
    ipam:
      config:
        - subnet: 192.168.70.0/24

x-deploy: &gpu-all-deploy
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: all
            capabilities: [gpu]

services:
  bert-score-api:
    image: ghcr.io/ilkersigirci/bert-score-api:latest
    container_name: bert-score-api
    restart: "no"
    <<: *gpu-all-deploy
    networks:
      - bert-score-api-net
    ports:
      - 8888:8888
    volumes:
      - YOUR_HF_HOME:/app/.cache/huggingface
    environment:
      - HF_HOME=/app/.cache/huggingface
      - HF_HUB_ENABLE_HF_TRANSFER=1
      - LANGUAGE=$LANGUAGE
      - RESCALE_WITH_BASELINE=$RESCALE_WITH_BASELINE
    healthcheck:
      test: "curl -f http://bert-score-api:8888/health"
      interval: 10s
      timeout: 5s
      retries: 3
```
