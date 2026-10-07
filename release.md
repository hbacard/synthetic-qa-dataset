# synthetic-qa-dataset v0.1.0

Generate a synthetic QA dataset from a set of documentation files, using any
OpenAI-compatible provider (OpenAI, Ollama, vLLM, ...), then validate that the
dataset is hallucination-free — all on top of [DeepEval](https://docs.confident-ai.com/).

## Highlights

- **Provider-agnostic**: works with any OpenAI-compatible API via `config.yaml`
  (custom `CustomOpenAI` + `CustomEmbeddingModel` in `providers.py`).
- **Hallucination check**: every generated Q/A record is validated against its
  context with DeepEval's `HallucinationMetric`.
- **Rate-limit handling**: automatic exponential-backoff retries on HTTP `429`,
  tuned per provider in `config.yaml`.
- **Two CLI entry points**: `synthetic-qa-generate` and `synthetic-qa-evaluate`.
- **Reproducible outputs**: timestamped results saved under `datasets/` and `evals/`.

## Install

```bash
pdm install
```

## Usage

```bash
pdm run synthetic-qa-generate
pdm run synthetic-qa-evaluate
```

See the [README](https://github.com/hbacard/synthetic-qa-dataset/blob/main/README.md)
for full CLI options and configuration.

## License

MIT · Python ≥ 3.12
