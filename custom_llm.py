from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI, AsyncOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

BASE_PROVIDER = "albert"
API_ENV_VAR = "NEW_ALBERT_API_KEY"
BASE_URL = "https://albert.api.etalab.gouv.fr/v1"


def init_client():
    return OpenAI(base_url=BASE_URL, api_key=os.environ.get(API_ENV_VAR, 'ollama'))

def init_async_client():
    return AsyncOpenAI(base_url=BASE_URL, api_key=os.environ.get(API_ENV_VAR, 'ollama'))


client = init_client()
a_client = init_async_client()

class CustomOpenAI(DeepEvalBaseLLM):
    def __init__(self, model):
        
        self.model = model 

    def load_model(self):
        return super().load_model(self.model)
    
    def generate(self, prompt: str):

        messages = [
            {'role': 'user', 'content': prompt}
        ]
        answer = client.chat.completions.create(
            model=self.model,
            temperature=0.7,
            messages=messages
        )

        return answer.choices[0].message.content

    async def a_generate(self, prompt: str):
            messages = [
                    {'role': 'user', 'content': prompt}
                ]
            answer = await a_client.chat.completions.create(
                model=self.model,
                temperature=0.7,
                messages=messages
            )
    
            return answer.choices[0].message.content

    def get_model_name(self):
        return "Custom OpenAI Model"


if __name__ == "__main__":
    import asyncio
    eval_llm = CustomOpenAI(model="openai/gpt-oss-120b")
    prompt = "What's the capital of Canada"
    res = asyncio.run(eval_llm.a_generate(prompt=prompt))
    print(res)

