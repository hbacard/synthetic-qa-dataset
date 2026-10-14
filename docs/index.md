# synthetic-qa-dataset

[![CI](https://img.shields.io/github/actions/workflow/status/hbacard/synthetic-qa-dataset/ci.yml?branch=main)](https://github.com/hbacard/synthetic-qa-dataset/actions)
[![License: MIT](https://img.shields.io/github/license/hbacard/synthetic-qa-dataset)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)

Generate a synthetic QA dataset from a set of documentation files, using your
preferred API (`https://your-preferred-provider/v1`) instead of OpenAI, then
validate that the dataset is hallucination-free.

The pipeline relies on [deepeval](https://docs.confident-ai.com/)'s
`generate_goldens_from_contexts`, connected to any OpenAI-compatible provider
via a custom LLM (defined in `providers.py`, with provider settings in
`config.yaml`). Each chunk is generated from with its source file attached, so
every record keeps its trace.

For installation, usage and configuration details, see the
[README](https://github.com/hbacard/synthetic-qa-dataset/blob/main/README.md).
