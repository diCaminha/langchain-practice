import json
import os
from dataclasses import Field

from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from openai import api_key
from pydantic import BaseModel
from pydantic.v1.json import pydantic_encoder


def busca_universidade_por_nome(nome: str):
    pass

def busca_universidades():
    pass


class UniversityName(BaseModel):
    name: str = Field("university name in lowercase")

class UniversityDataTool(BaseTool):
    name = "UniversityData"
    description = """
        This tool extract data about an university by given its name. 
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
        university_name = chain.invoke({"university_name": input})

        university = busca_universidade_por_nome(university_name)

        return json.dumps(university)