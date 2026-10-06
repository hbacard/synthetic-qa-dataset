"""Génère un jeu de données synthétique (goldens) pour l'évaluation RAG.

Pipeline : charge la configuration d'un provider, construit le modèle LLM et
l'embedder, génère des questions/réponses synthétiques à partir des documents
de ``docs/`` puis sauvegarde le résultat dans ``datasets/``.
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
    """Retourne les fichiers de ``docs_dir`` triés par nom."""
    if not docs_dir.is_dir():
        raise FileNotFoundError(
            f"Répertoire de documents introuvable : {docs_dir}"
        )
    paths = [p for p in docs_dir.iterdir() if p.is_file()]
    if not paths:
        raise FileNotFoundError(
            f"Aucun document trouvé dans {docs_dir} "
            "(fichiers .md, .txt, .pdf, .docx, ...)"
        )
    return sorted(paths)


def golden_to_dict(golden: Golden) -> dict:
    """Convertit un objet ``Golden`` deepeval en dictionnaire sérialisable."""
    return {
        "input": golden.input,
        "actual_output": getattr(golden, "actual_output", None),
        "expected_output": getattr(golden, "expected_output", None),
        "context": getattr(golden, "context", None),
        "metadata": getattr(golden, "metadata", None),
        "source_files": getattr(golden, "source_files", None),
    }


def save_goldens(goldens: list[Golden], output_dir: Path) -> Path:
    """Sauvegarde les goldens dans un fichier horodaté et renvoie son chemin."""
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
    LOGGER.info("Sauvegarde de %d goldens dans %s", len(goldens), file_path)
    return file_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère des goldens synthétiques depuis docs/."
    )
    parser.add_argument(
        "--provider",
        default="albert",
        help="Nom du provider dans config.yaml (défaut : albert).",
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=DOCS_DIR,
        help="Répertoire des documents sources (défaut : ./docs).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATASETS_DIR,
        help="Répertoire de sortie des goldens (défaut : ./datasets).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=1,
        help="Requêtes simultanées du Synthesizer (défaut : 1).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = get_provider(args.provider)
    custom_model = CustomOpenAI.from_provider(cfg)
    custom_embedder = CustomEmbeddingModel.from_provider(cfg)

    synthesizer = Synthesizer(model=custom_model, max_concurrent=args.max_concurrent)

    document_paths = [str(p) for p in list_document_paths(args.docs_dir)]
    LOGGER.info("Génération de goldens depuis %d document(s)...", len(document_paths))

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
        LOGGER.error("Échec de la génération des goldens : %s", exc)
        sys.exit(1)

    save_goldens(goldens, args.output_dir)

    for golden in goldens:
        LOGGER.info("INPUT : %s", golden.input)
        if golden.expected_output:
            LOGGER.info("EXPECTED : %s", golden.expected_output)
        LOGGER.info("---")


if __name__ == "__main__":
    main()
