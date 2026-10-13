# AGENTS.md

## Repo structure
├── AGENTS.md
├── CITATION.cff
├── codemeta.json
├── config.yaml.example
├── docs
│   ├── api.md
│   └── index.md
├── examples
│   └── test_hallucination_metric.py
├── LICENSE
├── mkdocs.yml
├── pdm.lock
├── pyproject.toml
├── README.md
├── src
│   └── synthetic_qa_dataset
│       ├── __init__.py
│       ├── config.py
│       ├── dataset_generator.py
│       ├── eval_synthetic_dataset.py
│       ├── paths.py
│       └── providers.py
├── tests
│   ├── test_config.py
│   ├── test_dataset_generator.py
│   ├── test_eval_synthetic_dataset.py
│   ├── test_paths.py
│   └── test_providers.py
└── uv.lock


## Setup commands
- Install deps: `pdm add` or `pdm pip install`
- Use the project virtual environment for all testing/running commands (e.g. `pdm run python ...`, `pdm run pytest ...`). Do not rely on system Python.

## Task management
- Divide long horizon tasks into milestones M1, M2, etc
- Request Commit approval after each milestone.

## Code style
- Add Numpy style docstring if necessary to functions
