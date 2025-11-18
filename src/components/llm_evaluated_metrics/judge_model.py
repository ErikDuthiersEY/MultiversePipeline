from deepeval.models.base_model import DeepEvalBaseLLM
from openai import AzureOpenAI

class AzureOpenAIJudgeModel(DeepEvalBaseLLM):
    def __init__(
        self,
        client:AzureOpenAI, 
        model_id:str
    ):
        self.client = client
        self.model_id = model_id

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return f"azure-openai-{self.model_id}" 