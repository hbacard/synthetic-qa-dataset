from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from dotenv import load_dotenv
import os
from custom_llm import CustomOpenAI, CustomEmbeddingModel

load_dotenv()


custom_model = CustomOpenAI(model="openai/gpt-oss-120b")
custom_embedder = CustomEmbeddingModel()

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

for g in goldens:
    print("INPUT:", g.input)
    if g.expected_output:
        print("EXPECTED:", g.expected_output)
    print("---")
