"""
Generates a synthetic QA dataset from documentation files.

Pipeline: loads the LLM provider and the embedding provider configurations,
builds the LLM model and the embedder, creates synthetic question/answer pairs
from the documents in ``docs/``, then saves the result in ``datasets/``.
"""

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from deepeval.dataset.golden import Golden
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig

from config import get_embedding_provider_name, get_llm_provider_name, get_provider
from paths import DATASETS_DIR, DOCS_DIR
from providers import CustomEmbeddingModel, CustomOpenAI

LOGGER = logging.getLogger(__name__)


def list_document_paths(docs_dir: Path) -> list[Path]:
    """Return the files in ``docs_dir`` sorted by name.

    Parameters
    ----------
    docs_dir:
        Directory whose files are to be listed.

    Returns
    -------
    list[Path]
        The files in ``docs_dir`` sorted by name.

    Raises
    ------
    FileNotFoundError
        If ``docs_dir`` does not exist or contains no files.
    """
    if not docs_dir.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")
    paths = [p for p in docs_dir.iterdir() if p.is_file()]
    if not paths:
        raise FileNotFoundError(
            f"No documents found in {docs_dir} " "(.md, .txt, .pdf, .docx, ... files)"
        )
    return sorted(paths)


def record_to_dict(golden: Golden) -> dict:
    """Convert a deepeval ``Golden`` object into a serializable dictionary.

    Parameters
    ----------
    golden:
        The deepeval golden record to convert.

    Returns
    -------
    dict
        Plain dictionary with the golden's fields ready for JSON serialization.
    """
    return {
        "input": golden.input,
        "actual_output": getattr(golden, "actual_output", None),
        "expected_output": getattr(golden, "expected_output", None),
        "context": getattr(golden, "context", None),
        "metadata": getattr(golden, "metadata", None),
        "source_files": getattr(golden, "source_files", None),
    }


def save_dataset(goldens: list[Golden], output_dir: Path) -> Path:
    """Save the QA records to a timestamped file and return its path.

    Parameters
    ----------
    goldens:
        The generated QA records to save.
    output_dir:
        Directory in which the dataset file is created.

    Returns
    -------
    Path
        Path of the written dataset file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    file_path = output_dir / f"synthetic_qa_{timestamp}.json"
    with open(file_path, "w", encoding="utf-8") as f_out:
        json.dump(
            [record_to_dict(g) for g in goldens],
            f_out,
            ensure_ascii=False,
            indent=2,
        )
    LOGGER.info("Saved %d records to %s", len(goldens), file_path)
    return file_path


def main() -> None:
    """Generate a synthetic QA dataset from ``docs/`` via the CLI."""
    parser = argparse.ArgumentParser(
        description="Generate a synthetic QA dataset from docs/."
    )
    parser.add_argument(
        "--llm-provider",
        default=get_llm_provider_name(),
        help="LLM provider name in config.yaml (default: %(default)s).",
    )
    parser.add_argument(
        "--embedding-provider",
        default=get_embedding_provider_name(),
        help="Embedding provider name in config.yaml (default: %(default)s).",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=DOCS_DIR,
        help="Directory of source documents (default: ./docs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATASETS_DIR,
        help="Output directory for the generated dataset (default: ./datasets).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Number of concurrent Synthesizer requests (default: 1).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    llm_cfg = get_provider(args.llm_provider)
    embedding_cfg = get_provider(args.embedding_provider)
    custom_model = CustomOpenAI.from_provider(llm_cfg)
    custom_embedder = CustomEmbeddingModel.from_provider(embedding_cfg)

    synthesizer = Synthesizer(model=custom_model, max_concurrent=args.max_concurrent)

    document_paths = [str(p) for p in list_document_paths(args.docs_dir)]
    LOGGER.info("Generating records from %d document(s)...", len(document_paths))

    try:
        goldens = synthesizer.generate_goldens_from_docs(
            document_paths=document_paths,
            include_expected_output=True,
            context_construction_config=ContextConstructionConfig(
                embedder=custom_embedder,
                critic_model=custom_model,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - catch-all for CLI entry point
        LOGGER.error("Failed to generate the dataset: %s", exc)
        sys.exit(1)

    save_dataset(goldens, args.output_dir)

    for golden in goldens:
        LOGGER.info("INPUT : %s", golden.input)
        if golden.expected_output:
            LOGGER.info("EXPECTED : %s", golden.expected_output)
        LOGGER.info("---")


if __name__ == "__main__":
    main()
