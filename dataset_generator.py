"""
Generates a synthetic dataset (goldens) for RAG evaluation.

Pipeline: loads a provider's configuration, builds the LLM model and the
embedder, creates synthetic question/answer pairs from the documents in
``docs/``, then saves the result in ``datasets/``.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from deepeval.synthesizer import Golden, Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig

from config import get_provider
from paths import DATASETS_DIR, DOCS_DIR
from providers import CustomEmbeddingModel, CustomOpenAI

LOGGER = logging.getLogger(__name__)


def list_document_paths(docs_dir: Path) -> list[Path]:
    """Return the files in ``docs_dir`` sorted by name."""
    if not docs_dir.is_dir():
        raise FileNotFoundError(
            f"Documents directory not found: {docs_dir}"
        )
    paths = [p for p in docs_dir.iterdir() if p.is_file()]
    if not paths:
        raise FileNotFoundError(
            f"No documents found in {docs_dir} "
            "(.md, .txt, .pdf, .docx, ... files)"
        )
    return sorted(paths)


def golden_to_dict(golden: Golden) -> dict:
    """Convert a deepeval ``Golden`` object into a serializable dictionary."""
    return {
        "input": golden.input,
        "actual_output": getattr(golden, "actual_output", None),
        "expected_output": getattr(golden, "expected_output", None),
        "context": getattr(golden, "context", None),
        "metadata": getattr(golden, "metadata", None),
        "source_files": getattr(golden, "source_files", None),
    }


def save_goldens(goldens: list[Golden], output_dir: Path) -> Path:
    """Save the goldens to a timestamped file and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = output_dir / f"goldens_{timestamp}.json"
    with open(file_path, "w", encoding="utf-8") as f_out:
        json.dump(
            [golden_to_dict(g) for g in goldens],
            f_out,
            ensure_ascii=False,
            indent=2,
        )
    LOGGER.info("Saved %d goldens to %s", len(goldens), file_path)
    return file_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic goldens from docs/."
    )
    parser.add_argument(
        "--provider",
        default="albert",
        help="Provider name in config.yaml (default: albert).",
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
        help="Output directory for the goldens (default: ./datasets).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Number of concurrent Synthesizer requests (default: 1).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = get_provider(args.provider)
    custom_model = CustomOpenAI.from_provider(cfg)
    custom_embedder = CustomEmbeddingModel.from_provider(cfg)

    synthesizer = Synthesizer(model=custom_model, max_concurrent=args.max_concurrent)

    document_paths = [str(p) for p in list_document_paths(args.docs_dir)]
    LOGGER.info("Generating goldens from %d document(s)...", len(document_paths))

    try:
        goldens = synthesizer.generate_goldens_from_docs(
            document_paths=document_paths,
            include_expected_output=True,
            context_construction_config=ContextConstructionConfig(
                embedder=custom_embedder,
                critic_model=custom_model,
            ),
        )
    except Exception as exc:
        LOGGER.error("Failed to generate goldens: %s", exc)
        sys.exit(1)

    save_goldens(goldens, args.output_dir)

    for golden in goldens:
        LOGGER.info("INPUT : %s", golden.input)
        if golden.expected_output:
            LOGGER.info("EXPECTED : %s", golden.expected_output)
        LOGGER.info("---")


if __name__ == "__main__":
    main()
