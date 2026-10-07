# synthetic-qa-dataset

Generate a synthetic QA dataset from a set of documentation files, using your
preferred API (`https://your-preferred-provider/v1`) instead of OpenAI, then
validate that the dataset is hallucination-free.

The pipeline relies on [deepeval](https://docs.confident-ai.com/)'s
`generate_goldens_from_docs`, connected to any OpenAI-compatible provider via a
custom LLM and a custom embedding model (both defined in `providers.py`, with
provider settings in `config.yaml`).

For installation, usage and configuration details, see the
[README](https://github.com/hbacard/synthetic-qa-dataset/blob/main/README.md).
