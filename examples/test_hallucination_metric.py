from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from synthetic_qa_dataset.config import get_llm_provider_name, get_provider
from synthetic_qa_dataset.providers import CustomOpenAI

load_dotenv()


context = [
    (
        "Supervised machine learning trains a model on labeled data, where each example has input features and a known target label. "
        "The dataset is typically split into training and test sets: the model learns patterns from the training set and is then "
        "evaluated on the unseen test set. Common evaluation metrics include accuracy, precision, recall, and F1-score. "
        "Overfitting occurs when a model memorizes the training data but fails to generalize to new data."
    )
]

actual_output = """
The supervised learning workflow starts with a labeled dataset, which is divided into a training set and a test set.
The model is trained on the training data and later assessed on the test data using metrics such as accuracy,
precision, recall, and F1-score. Regularization helps prevent overfitting, so the model generalizes well to new examples.
"""


test_case = LLMTestCase(
    input="Explain the supervised machine learning workflow and how model performance is evaluated.",
    actual_output=actual_output,
    context=context,
)

eval_llm = CustomOpenAI.from_provider(get_provider(get_llm_provider_name()))
halu_metric = HallucinationMetric(threshold=0.5, model=eval_llm)

##

halu_metric.measure(test_case)
print("Hallucination Metric:")
print("  Score: ", halu_metric.score)
print("  Reason: ", halu_metric.reason)
