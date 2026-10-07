# synthetic-qa-dataset

Generate a synthetic QA dataset from a set of documentation files, using your
preferred API (`https://your-preferred-provider/v1`) instead of OpenAI, then
validate that the dataset is hallucination-free.

The pipeline relies on [deepeval](https://docs.confident-ai.com/)'s
`generate_goldens_from_docs`, connected to any OpenAI-compatible provider via a
custom LLM and a custom embedding model (both defined in `providers.py`, with
provider settings in `config.yaml`).

## Prerequisites

- Python 3.12 with `uv` installed.
- An API key for your provider in the environment. The env variable name is
  set per provider in `config.yaml` (`api_env`). For Albert, e.g.:

  ```bash
  export PROVIDER_API_KEY="your-key-here"
  ```

  If the key is missing, requests fall back to the literal string `ollama`,
  which is only useful when talking to a local Ollama server.

## Installation

```bash
uv sync
```

This installs all dependencies (deepeval, openai, pydantic, chromadb,
langchain, pypdf, docx2txt, ...) into a local `.venv`.

## Documents

Put the source documentation inside `./docs/` as Markdown (or any format
supported by deepeval: `.md`, `.txt`, `.pdf`, `.docx`, ...). Each file becomes
one source document that is chunked into contexts.

`dataset_generator.py` automatically picks up every **file** directly inside
`./docs/` (it does not recurse into subdirectories).

## Running

Generate the dataset:

```bash
uv run python dataset_generator.py
```

Evaluate a generated dataset (checks each `expected_output` is grounded in its
`context`):

```bash
uv run python eval_synthetic_dataset.py
```

### Dataset generation: step by step

1. `dataset_generator.py` loads the LLM provider and the embedding provider from
   `config.yaml` (via `config.py`) and builds a `CustomOpenAI` (the
   generation/critic model) from the LLM provider and a `CustomEmbeddingModel`
   (the embedder) from the embedding provider.
2. It lists every file in `./docs/` and passes that list to
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

`dataset_generator.py`:

| Option                | Default                      | Purpose                                    |
| --------------------- | ---------------------------- | ------------------------------------------ |
| `--llm-provider`      | `llm_provider` in config.yaml| LLM (generation/critic) provider.          |
| `--embedding-provider`| `embedding_provider` in config.yaml | Embedding provider.                |
| `--docs-dir`          | `./docs`                     | Directory containing the source docs.      |
| `--output-dir`        | `./datasets`                 | Where to save the generated dataset.       |
| `--max-concurrent`    | `1`                          | Concurrent Synthesizer requests (serial).  |

`eval_synthetic_dataset.py`:

| Option         | Default                     | Purpose                                  |
| -------------- | --------------------------- | ---------------------------------------- |
| `--dataset`    | most recent `synthetic_qa_*.json`| Dataset file name in `datasets/`.        |
| `--llm-provider` | `llm_provider` in config.yaml | LLM (critic) provider.                 |
| `--threshold`  | `0.8`                       | Hallucination metric threshold.          |
| `--output-dir` | `./evals`                   | Where to save the evaluation results.    |

Example with a local Ollama server:

```bash
uv run python dataset_generator.py --llm-provider ollama --embedding-provider ollama
uv run python eval_synthetic_dataset.py --llm-provider ollama --threshold 0.9
```

## Handling the rate limit

The Albert API allows about **10 requests/minute**. A large run (several
documents × several contexts × several QA records each) easily exceeds this and
Albert returns HTTP `429`.

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
from config import get_provider, get_llm_provider_name, get_embedding_provider_name
from providers import CustomOpenAI, CustomEmbeddingModel

llm_cfg = get_provider(get_llm_provider_name())
embedding_cfg = get_provider(get_embedding_provider_name())

llm = CustomOpenAI.from_provider(llm_cfg)          # chat / critic model
embedder = CustomEmbeddingModel.from_provider(embedding_cfg)  # embedding model
```

If `llm_provider` / `embedding_provider` are missing, both fall back to a single
provider (the first configured one, or `albert`) so legacy configs keep working.

The constructors also accept each setting as an explicit argument, so any
OpenAI-compatible provider (OpenAI, Ollama, vLLM, Azure OpenAI-compatible
endpoints, ...) works with the same code by adding another entry to
`config.yaml` and calling `get_provider("<name>")`.

## Project layout

```
config.yaml       # per-provider settings + llm_provider / embedding_provider
config.py         # loads config.yaml, exposes get_provider + provider selectors
providers.py      # CustomOpenAI + CustomEmbeddingModel (provider-agnostic)
paths.py          # shared project paths (docs/, datasets/, evals/)
dataset_generator.py      # generate synthetic QA dataset + auto-save
eval_synthetic_dataset.py # validate dataset for hallucination
docs/             # input documentation files
datasets/         # generated synthetic QA datasets (git-ignored, timestamped files)
evals/            # evaluation results (git-ignored)
```