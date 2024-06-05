import yaml
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI


class LLMCaller:


    def __init__(self):
        with open("data/params/secret.yaml", "r") as file:
            self.api_key = yaml.safe_load(file)['api_key']


    def call_llm(self, prompt: str, data: dict, model_name: str = "gpt-4-turbo",  temperature: float = 0.0) -> str:

        formatted_prompt = prompt.format(**data)
        print("@@@@@@@@@@@@@@@@@@@@@")
        print(formatted_prompt)
        print("@@@@@@@@@@@@@@@@@@@@@")

        llm = ChatOpenAI(model=model_name, openai_api_key=self.api_key)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        results_raw = llm_chain.run(temperature=temperature, **data)

        print("**************************")
        print(results_raw)
        print("**************************")

        return results_raw
