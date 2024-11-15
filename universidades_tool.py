import json
import os

from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from pydantic import BaseModel
import pandas as pd


def busca_universidade_por_nome(nome: str):
    dados = pd.read_csv("docs/universidades.csv")
    dados_universidade = dados[dados["NOME_FACULDADE"] == nome]
    if dados_universidade.empty:
        return {}
    return dados_universidade[:1].to_dict()

def busca_universidades():
    pass


class UniversityName(BaseModel):
    name: str = Field("university name in lowercase")


class UniversityDataTool(BaseTool):
    name = "UniversityData"
    description = """
        This tool extract data and informations about an university by given its name. 
        Pass to this tool as input the name of the university.
    """

    def _run(self, input: str) -> str:
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPEN_AI_KEY")
        )

        parser = JsonOutputParser(pydantic_object = UniversityName)

        template = PromptTemplate(
            template = """
                You must analyze the input name of the university, format to lowercase, and extact only 
                the name of the university.
                
                Input:
                --------------------
                {input}
                --------------------
                
                Output Format:
                {output_format}
            """,
            input_variables=["input"],
            partial_variables={"output_format": parser.get_format_instructions()}
        )

        chain = template | llm | parser
        university_name = chain.invoke({"input": input})

        university = busca_universidade_por_nome(university_name)

        return json.dumps(university)