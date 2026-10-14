"""
Generates a synthetic QA dataset from documentation files.

Pipeline: loads the LLM provider configuration, builds the LLM model, chunks
the documents in ``corpus/`` into contexts, creates synthetic question/answer
pairs with deepeval, then saves the result in ``datasets/``.

Unlike ``generate_goldens_from_docs`` (which loses the source link), each
document chunk is passed as its own context together with its source file, so
every generated record keeps its trace via ``source_file`` and
``additional_metadata`` (``context_source_files``, ``used_source_files``).
"""

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from deepeval.dataset.golden import Golden
from deepeval.synthesizer import Synthesizer
from langchain_text_splitters import TokenTextSplitter

from .config import get_llm_provider_name, get_provider
from .paths import CORPUS_DIR, DATASETS_DIR
from .providers import CustomOpenAI

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
            f"No documents found in {docs_dir} (.md, .txt, .pdf, .docx, ... files)"
        )
    return sorted(paths)


def read_document_text(document_path: Path) -> str:
    """Read the text content of a corpus document.

    Supports plain text (``.txt``, ``.md``, ``.markdown``), PDF (``.pdf``)
    and Word (``.docx``) files.

    Parameters
    ----------
    document_path:
        Path of the document to read.

    Returns
    -------
    str
        The full text content of the document.

    Raises
    ------
    ImportError
        If a binary format (``.pdf``/``.docx``) is requested without its
        optional dependency installed.
    """
    suffix = document_path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError("pypdf is required to read PDF documents") from exc
        reader = PdfReader(str(document_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        try:
            import docx2txt
        except ImportError as exc:
            raise ImportError("docx2txt is required to read .docx documents") from exc
        return docx2txt.process(str(document_path))
    try:
        return document_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return document_path.read_text(encoding="latin-1")


def build_contexts(
    document_paths: list[Path], chunk_size: int = 1024, chunk_overlap: int = 0
) -> tuple[list[list[str]], list[str]]:
    """Chunk the documents and return the contexts and their source files.

    Each chunk becomes its own context so that every generated record can be
    traced back to the exact source file it was produced from.

    Parameters
    ----------
    document_paths:
        Documents to chunk.
    chunk_size:
        Maximum number of tokens per chunk.
    chunk_overlap:
        Number of overlapping tokens between consecutive chunks.

    Returns
    -------
    tuple[list[list[str]], list[str]]
        ``contexts`` where each element is a single-chunk context, and
        ``source_files`` with the matching source document per context.
    """
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    contexts: list[list[str]] = []
    source_files: list[str] = []
    for path in document_paths:
        for chunk in splitter.split_text(read_document_text(path)):
            contexts.append([chunk])
            source_files.append(str(path))
    return contexts, source_files


def record_to_dict(golden: Golden) -> dict:
    """Convert a deepeval ``Golden`` object into a serializable dictionary.

    The trace is preserved from the golden's ``source_file`` field and its
    ``additional_metadata`` (``context_source_files``, ``used_source_files``,
    ``synthetic_input_quality``, ``evolutions``).

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
        "id": getattr(golden, "id", None),
        "input": golden.input,
        "actual_output": getattr(golden, "actual_output", None),
        "expected_output": getattr(golden, "expected_output", None),
        "context": getattr(golden, "context", None),
        "source_file": getattr(golden, "source_file", None),
        "additional_metadata": getattr(golden, "additional_metadata", None),
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
    """Generate a synthetic QA dataset from ``corpus/`` via the CLI."""
    parser = argparse.ArgumentParser(
        description="Generate a synthetic QA dataset from corpus/."
    )
    parser.add_argument(
        "--llm-provider",
        default=get_llm_provider_name(),
        help="LLM provider name in config.yaml (default: %(default)s).",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=CORPUS_DIR,
        help="Directory of source documents (default: ./corpus).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATASETS_DIR,
        help="Output directory for the generated dataset (default: ./datasets).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Maximum number of tokens per chunk (default: 1024).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=0,
        help="Number of overlapping tokens between chunks (default: 0).",
    )
    parser.add_argument(
        "--max-goldens-per-context",
        type=int,
        default=2,
        help="Question/answer records generated per context (default: 2).",
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
    custom_model = CustomOpenAI.from_provider(llm_cfg)

    synthesizer = Synthesizer(model=custom_model, max_concurrent=args.max_concurrent)

    document_paths = list_document_paths(args.corpus_dir)
    contexts, source_files = build_contexts(
        document_paths, args.chunk_size, args.chunk_overlap
    )
    LOGGER.info(
        "Generating records from %d context(s) across %d document(s)...",
        len(contexts),
        len(document_paths),
    )

    try:
        goldens = synthesizer.generate_goldens_from_contexts(
            contexts=contexts,
            source_files=source_files,
            include_expected_output=True,
            max_goldens_per_context=args.max_goldens_per_context,
        )
    except Exception as exc:  # noqa: BLE001 - catch-all for CLI entry point
        LOGGER.error("Failed to generate the dataset: %s", exc)
        sys.exit(1)

    save_dataset(goldens, args.output_dir)

    for golden in goldens:
        LOGGER.info("INPUT : %s", golden.input)
        if golden.expected_output:
            LOGGER.info("EXPECTED : %s", golden.expected_output)
        if golden.source_file:
            LOGGER.info("SOURCE : %s", golden.source_file)
        LOGGER.info("---")


if __name__ == "__main__":
    main()
