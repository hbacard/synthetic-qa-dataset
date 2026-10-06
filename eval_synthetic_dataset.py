import os
from deepeval import evaluate
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from providers import CustomOpenAI
from config import get_provider
from pathlib import Path
import json
load_dotenv()


ROOT_DIR = Path(__file__).parent 
DATASETS_DIR = ROOT_DIR / "datasets"
EVALS_DIR = ROOT_DIR / "evals"

def load_data(dataset_name: str):
    dataset_path = DATASETS_DIR.joinpath(dataset_name)
    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_name} not found")
    with open(dataset_path, 'r') as f_in:
        data = json.load(f_in)
    return data


def eval_synthetic_data(dataset_name: str, provider: str = "ollama"):
    eval_llm = CustomOpenAI.from_provider(get_provider(provider))
    halu_metric = HallucinationMetric(threshold=0.8, model=eval_llm) # For hallucination
    data : list[dict] = load_data(dataset_name=dataset_name)
    results = []
    for _, d in enumerate(data):
        test_case = LLMTestCase(
            input=d.get('input', ""),
            actual_output=d.get("expected_output", ""),
            context=d.get("context",""),
            )
        # run model to judge

        halu_metric.measure(test_case)

        results.append({"halucination_metric": {"score": halu_metric.score, "reason": halu_metric.reason}})
    # saving
    output_path = EVALS_DIR.joinpath(f"eval_data_{dataset_name}")
    with open(output_path, "w+") as f_out:
        json.dump(results, f_out, indent=2)
        print("Eval dataset saved at")
        
    return results
        

if __name__ == "__main__":
    dataset_name = "goldens_20261006_133510.json"
    provider = "ollama"
    results = eval_synthetic_data(dataset_name=dataset_name, provider=provider)

