"""Évalue un jeu de données synthétiques avec la métrique d'hallucination.

Charge un dataset de goldens généré par ``dataset_generator.py`` et vérifie que
chaque ``expected_output`` est fidèle au ``context`` associé (validation des
goldens, pas un test de pipeline RAG). Les résultats sont sauvegardés dans
``evals/``.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase

from config import get_provider
from paths import DATASETS_DIR, EVALS_DIR
from providers import CustomOpenAI

LOGGER = logging.getLogger(__name__)


def latest_dataset(datasets_dir: Path) -> Path:
    """Renvoie le fichier ``goldens_*.json`` le plus récent du répertoire."""
    candidates = sorted(datasets_dir.glob("goldens_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"Aucun dataset goldens_*.json trouvé dans {datasets_dir}"
        )
    return candidates[-1]


def load_data(dataset_path: Path) -> list[dict]:
    """Charge un dataset JSON et vérifie qu'il s'agit d'une liste."""
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset introuvable : {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f_in:
        data = json.load(f_in)
    if not isinstance(data, list):
        raise ValueError(
            f"Format invalide pour {dataset_path} : une liste est attendue"
        )
    return data


def build_test_case(record: dict) -> LLMTestCase:
    """Construit un LLMTestCase à partir d'un enregistrement du dataset.

    ``actual_output`` prend la valeur d'``expected_output`` pour valider que le
    golden généré est bien fidèle au contexte (pas d'hallucination).
    """
    return LLMTestCase(
        input=record.get("input", ""),
        actual_output=record.get("expected_output", ""),
        context=record.get("context") or [],
    )


def evaluate_dataset(
    dataset_name: str, provider: str, threshold: float
) -> list[dict]:
    """Mesure l'hallucination de chaque golden et renvoie les résultats."""
    eval_llm = CustomOpenAI.from_provider(get_provider(provider))
    halu_metric = HallucinationMetric(threshold=threshold, model=eval_llm)

    data = load_data(DATASETS_DIR / dataset_name)
    results: list[dict] = []
    failures = 0

    for index, record in enumerate(data, start=1):
        test_case = build_test_case(record)
        try:
            halu_metric.measure(test_case)
        except Exception as exc:
            failures += 1
            LOGGER.error(
                "Échec de l'évaluation de l'élément %d/%d : %s",
                index,
                len(data),
                exc,
            )
            continue

        results.append(
            {
                "hallucination_metric": {
                    "score": halu_metric.score,
                    "reason": halu_metric.reason,
                }
            }
        )
        LOGGER.info(
            "Élément %d/%d évalué (score : %.2f)",
            index,
            len(data),
            halu_metric.score,
        )

    if failures:
        LOGGER.warning("%d élément(s) en échec sur %d", failures, len(data))
    return results


def save_results(results: list[dict], dataset_name: str, output_dir: Path) -> Path:
    """Sauvegarde les résultats d'évaluation et renvoie le chemin du fichier."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"eval_data_{dataset_name}.json"
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=2)
    LOGGER.info("Résultats sauvegardés dans %s", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Évalue l'hallucination des goldens d'un dataset."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Nom du fichier de dataset (défaut : le plus récent de datasets/).",
    )
    parser.add_argument(
        "--provider",
        default="albert",
        help="Nom du provider dans config.yaml (défaut : albert).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Seuil de la métrique d'hallucination (défaut : 0.8).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EVALS_DIR,
        help="Répertoire de sortie des évaluations (défaut : ./evals).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dataset_name = args.dataset or latest_dataset(DATASETS_DIR).name
    LOGGER.info("Évaluation du dataset %s...", dataset_name)

    try:
        results = evaluate_dataset(
            dataset_name=dataset_name,
            provider=args.provider,
            threshold=args.threshold,
        )
    except Exception as exc:
        LOGGER.error("Échec de l'évaluation : %s", exc)
        sys.exit(1)

    save_results(results, dataset_name, args.output_dir)


if __name__ == "__main__":
    main()
