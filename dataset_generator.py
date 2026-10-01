from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from dotenv import load_dotenv
from datetime import datetime
import os
import json
from providers import CustomOpenAI, CustomEmbeddingModel
from config import get_provider

load_dotenv()

OUTPUT_DIR = "./datasets"


def golden_to_dict(g: object) -> dict:
    return {
        "input": g.input,
        "actual_output": getattr(g, "actual_output", None),
        "expected_output": getattr(g, "expected_output", None),
        "context": getattr(g, "context", None),
        "metadata": getattr(g, "metadata", None),
        "source_files": getattr(g, "source_files", None),
    }


def save_goldens(goldens: list, directory: str = OUTPUT_DIR) -> str:
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(directory, f"goldens_{timestamp}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump([golden_to_dict(g) for g in goldens], f, ensure_ascii=False, indent=2)
    print(f"Saved {len(goldens)} goldens to {file_path}")
    return file_path


cfg = get_provider("albert")
custom_model = CustomOpenAI.from_provider(cfg)
custom_embedder = CustomEmbeddingModel.from_provider(cfg)

synthesizer = Synthesizer(model=custom_model, max_concurrent=1)

document_paths = [
    os.path.join("./docs", f)
    for f in os.listdir("./docs")
    if os.path.isfile(os.path.join("./docs", f))
]

goldens = synthesizer.generate_goldens_from_docs(
    document_paths=document_paths,
    include_expected_output=True,
    context_construction_config=ContextConstructionConfig(
        embedder=custom_embedder,
        critic_model=custom_model,
    ),
)

save_goldens(goldens)

for g in goldens:
    print("INPUT:", g.input)
    if g.expected_output:
        print("EXPECTED:", g.expected_output)
    print("---")
