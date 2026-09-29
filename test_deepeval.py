import os
from deepeval import evaluate
from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

load_dotenv()



context = [
    "The Great Wall of China is a series of fortifications made of stone, brick, tamped earth, wood, and other materials, "
    "generally built along an east-to-west line across the historical northern borders of China to protect the Chinese states "
    "and empires against the raids and invasions of the nomadic groups of the Eurasian Steppe."
]

actual_output = ("The Great Wall of China is made entirely of gold and was built in a single year by the Ming Dynasty to store treasures.")

# 

test_case = LLMTestCase(
    input="What is the Great Wall of China made of and why was it built?",
    actual_output=actual_output,
    context=context,

)

halu_metric = HallucinationMetric(threshold=0.5)

## 

halu_metric.measure(test_case)
print("Hallucination Metric:")
print("  Score: ", halu_metric.score)
print("  Reason: ", halu_metric.reason)
