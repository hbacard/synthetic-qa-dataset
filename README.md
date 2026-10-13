# Synthetic QA Dataset Generator

[![CI](https://img.shields.io/github/actions/workflow/status/hbacard/synthetic-qa-dataset/ci.yml?branch=main)](https://github.com/hbacard/synthetic-qa-dataset/actions)
[![License: MIT](https://img.shields.io/github/license/hbacard/synthetic-qa-dataset)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)

Build a synthetic question-answering dataset from a collection of text files:

1. **Generate**: use a model served via vLLM or Ollama (OpenAI-compatible endpoint, e.g. `http://localhost:8000/v1`) rather than the OpenAI API.
2. **Validate**: verify that every QA pair is supported by its source text, filtering out any hallucinated content.

The pipeline relies on [deepeval](https://docs.confident-ai.com/)'s
`generate_goldens_from_docs`, connected to any OpenAI-compatible provider via a
custom LLM and a custom embedding model (both defined in `providers.py`, with
provider settings in `config.yaml`).

## Quick start

Run through this once and you'll have a generated and validated dataset:

1. **Install Python 3.12 and `pdm`.** Follow the instructions to install [pdm](https://github.com/pdm-project/pdm)

2. **Create your configuration.** `config.yaml` is git-ignored, so copy the
   example and edit it to match your provider:

   ```bash
   cp config.yaml.example config.yaml
   ```

   In `config.yaml`, set your provider's `model`, `base_url` and `api_env`, and
   the `llm_provider` / `embedding_provider` keys.

3. **Create your environment file.** Copy `.env.example` to `.env` and put your
   real API key in it. The variable name must match the `api_env` field of the
   provider you use in `config.yaml`:

   ```bash
   cp .env.example .env
   # edit .env, e.g.: OLLAMA_API_KEY=your-key-here
   ```

   The file is read automatically by `python-dotenv` — no `export` needed.

4. **Install the dependencies** into a local virtual environment:

   ```bash
   pdm install
   ```

5. **Add your source documentation.** Create the `corpus/` directory and put
   your source files inside it (`.md`, `.txt`, `.pdf`, `.docx`, ...):

   ```bash
   mkdir corpus
   ```

6. **Generate the dataset:**


    ```bash
    pdm run synthetic-qa-generate
    ```

7. **Evaluate the generated dataset**
    ```bash
    pdm run synthetic-qa-evaluate
    ```


## Documents

Put the source documentation inside `./corpus/` as Markdown (or any format
supported by deepeval: `.md`, `.txt`, `.pdf`, `.docx`, ...). Each file becomes
one source document that is chunked into contexts.

`dataset_generator.py` automatically picks up every **file** directly inside
`./corpus/` (it does not recurse into subdirectories).

## Dataset generation: step by step explaination

1. `dataset_generator.py` loads the LLM provider and the embedding provider from
   `config.yaml` (via `config.py`) and builds a `CustomOpenAI` (the
   generation/critic model) from the LLM provider and a `CustomEmbeddingModel`
   (the embedder) from the embedding provider.
2. It lists every file in `./corpus/` and passes that list to
   `generate_goldens_from_docs`.
3. `generate_goldens_from_docs` chunks each document into contexts (using the
   embedder), then generates a set of synthetic question/answer records.
4. The resulting records are **automatically saved** to
   `./datasets/synthetic_qa_<YYYYMMDD_HHMMSS>.json` (timestamped filename).

### Dataset evaluation

`eval_synthetic_dataset.py` loads a synthetic dataset and, for each record,
builds an `LLMTestCase` where `actual_output = expected_output` (the generated
answer) is judged against its `context` with deepeval's `HallucinationMetric`.
This validates that the synthetic dataset is hallucination-free; it is **not** a
test of a RAG pipeline. Results are saved to
`./evals/eval_data_<dataset_name>.json`.

By default it evaluates the **most recent** `synthetic_qa_*.json` file in
`datasets/`.

### Output

- **Dataset generation**: `./datasets/synthetic_qa_<timestamp>.json` containing
  the full QA records. `./datasets/` is git-ignored.
- **Evaluation**: `./evals/eval_data_<dataset_name>.json`, one
  `hallucination_metric` (score + reason) per record. `./evals/` is git-ignored.

### CLI options

`synthetic-qa-generate` (aka `python -m synthetic_qa_dataset.dataset_generator`):

| Option                | Default                      | Purpose                                    |
| --------------------- | ---------------------------- | ------------------------------------------ |
| `--llm-provider`      | `llm_provider` in config.yaml| LLM (generation/critic) provider.          |
| `--embedding-provider`| `embedding_provider` in config.yaml | Embedding provider.                |
| `--corpus-dir`        | `./corpus`                   | Directory containing the source docs.      |
| `--output-dir`        | `./datasets`                 | Where to save the generated dataset.       |
| `--max-concurrent`    | `1`                          | Concurrent Synthesizer requests (serial).  |

`synthetic-qa-evaluate` (aka `python -m synthetic_qa_dataset.eval_synthetic_dataset`):

| Option         | Default                     | Purpose                                  |
| -------------- | --------------------------- | ---------------------------------------- |
| `--dataset`    | most recent `synthetic_qa_*.json`| Dataset file name in `datasets/`.        |
| `--llm-provider` | `llm_provider` in config.yaml | LLM (critic) provider.                 |
| `--threshold`  | `0.8`                       | Hallucination metric threshold.          |
| `--output-dir` | `./evals`                   | Where to save the evaluation results.    |

Example with a local Ollama server:

```bash
pdm run synthetic-qa-generate --llm-provider ollama --embedding-provider ollama
pdm run synthetic-qa-evaluate --llm-provider ollama --threshold 0.9
```

## Handling the rate limit

A tested API allows about **10 requests/minute**. A large run (several
documents × several contexts × several QA records each) easily exceeds this and
some providers return HTTP `429`.

`providers.py` handles this automatically:

- Every chat and embedding call is wrapped in a retry loop.
- On HTTP `429` (rate limit) or connection/timeout errors, it waits with
  **exponential backoff** (`6s → 12s → 24s → ...`, capped at 60s) and retries,
  up to `DEFAULT_MAX_RETRIES` (8) attempts.
- After exhausting retries it raises `RequestLimitError` with a clear message.

Tunables are set per provider in `config.yaml`:

| Constant              | Default | Purpose                                |
| --------------------- | ------- | -------------------------------------- |
| `DEFAULT_MAX_RETRIES` | 8       | Max attempts per call (incl. first).   |
| `DEFAULT_MIN_SLEEP`   | 6.0     | Initial backoff delay in seconds.      |
| `DEFAULT_MAX_DELAY`   | 60.0    | Cap on the backoff delay in seconds.   |

`dataset_generator.py` also sets `max_concurrent=1` on the `Synthesizer` so requests
are sent serially rather than in parallel.

If a run still fails after all retries, re-run `dataset_generator.py` — it will resume
from the (unchanged) documents and produce a new timestamped dataset.

## Configuration

Providers are configured in `config.yaml` and loaded by `config.py`. Each
provider is keyed by name with the following optional fields:

```yaml
providers:
  your_provider:
    model: model_name   # generation / critic model
    embedding: bge-m3            # embedding model
    base_url: https://your-endpoint/v1
    api_env: "PROVIDER_API_KEY"      # env var holding the API key
    temperature: 0.7             # generation temperature (default 0.7)
    max_retries: 8               # max attempts per call (default 8)
    min_sleep: 6.0               # initial backoff seconds (default 6.0)
    max_delay: 60.0              # backoff cap, seconds (default 60.0)

# Which provider is used for the LLM and which for the embeddings. They can
# differ (e.g. LLM from one API, embeddings from another).
llm_provider: your_provider
embedding_provider: your_provider
```

`config.py` exposes `get_provider(name)` returning a `ProviderConfig`
dataclass, plus `get_llm_provider_name()` and `get_embedding_provider_name()`
returning the providers selected by the top-level keys. Instantiate the clients
from the configs directly, or via the `from_provider` classmethods:

```python
from synthetic_qa_dataset.config import (
    get_provider,
    get_llm_provider_name,
    get_embedding_provider_name,
)
from synthetic_qa_dataset.providers import CustomOpenAI, CustomEmbeddingModel

llm_cfg = get_provider(get_llm_provider_name())
embedding_cfg = get_provider(get_embedding_provider_name())

llm = CustomOpenAI.from_provider(llm_cfg)  # chat / critic model
embedder = CustomEmbeddingModel.from_provider(embedding_cfg)  # embedding model
```

Run the snippet from the repo root with the package installed.

If `llm_provider` / `embedding_provider` are missing, both fall back to a single
provider (the first configured one, or `ollama`) so legacy configs keep working.

The constructors also accept each setting as an explicit argument, so any
OpenAI-compatible provider (OpenAI, Ollama, vLLM, Azure OpenAI-compatible
endpoints, ...) works with the same code by adding another entry to
`config.yaml` and calling `get_provider("<name>")`.

## Troubleshooting

- **`KeyError: Provider 'ollama' not found`** — `config.yaml` is missing or the
  provider name doesn't match. It is git-ignored, so on a fresh clone run
  `cp config.yaml.example config.yaml` and check the provider name.

- **API key not picked up** — make sure `.env` exists (copy it with
  `cp .env.example .env`), the variable name matches your provider's `api_env`,
  and re-copy/edit `.env` after changing providers.

- **HTTP 429 (rate limit) errors** — `providers.py` retries automatically with
  exponential backoff (see [Handling the rate limit](#handling-the-rate-limit)).
  If a run still fails, re-run `synthetic-qa-generate` — it produces a new
  timestamped dataset.

- **Ollama not reachable** — start the server first (`ollama serve`) and check
  that `base_url` in `config.yaml` points to `http://localhost:11434/v1`.

Note: `config.yaml`, `.env`, `corpus/`, `datasets/` and `evals/` are all
git-ignored, so they only exist locally after a fresh clone.

## Project layout

```
config.yaml       # per-provider settings + llm_provider / embedding_provider
src/
  synthetic_qa_dataset/  # the package
    config.py            # loads config.yaml, exposes get_provider + selectors
    providers.py         # CustomOpenAI + CustomEmbeddingModel (provider-agnostic)
    paths.py             # shared project paths (corpus/, datasets/, evals/)
    dataset_generator.py # generate synthetic QA dataset + auto-save
    eval_synthetic_dataset.py # validate dataset for hallucination
examples/
  test_hallucination_metric.py # standalone metric demo script
corpus/           # input documentation files (git-ignored)
datasets/         # generated synthetic QA datasets (git-ignored, timestamped files)
evals/            # evaluation results (git-ignored)
```