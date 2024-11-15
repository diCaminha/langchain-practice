import json
import os
from tempfile import template

from langchain.tools import BaseTool
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd


def busca_dados_de_estudante(estudante):
    dados = pd.read_csv("docs/estudantes.csv")
    dados_com_esse_estudante = dados[dados["USUARIO"] == estudante]
    if dados_com_esse_estudante.empty:
        return {}
    return dados_com_esse_estudante.iloc[:1].to_dict()


class StudentExtractor(BaseModel):
    name: str = Field("student first name, always in lowercase.")


class StudentData(BaseTool):
    name = "StudentData"
    description = "This tool is responsable for extract information and history data of the user."

    def _run(self, input: str) -> str :
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPEN_AI_KEY")
        )

        parser = JsonOutputParser(pydantic_object=StudentExtractor)

        template = PromptTemplate(
            template="""You should analyze the {input} info and extract the name informed.
            Output format:
            {output_format}
            """,
            input_variables=["input"],
            partial_variables={"output_format": parser.get_format_instructions()})

        cadeia = template | llm | parser

        student_name = cadeia.invoke({"input": input})
        dados = busca_dados_de_estudante(student_name["name"])
        return json.dumps(dados)


class AcademicProfile(BaseModel):
    name: str = Field("name of the student")
    conclusion_year = Field("the year where the student will finish the graduation")
    summary: str = Field("summary of the main characteristics of this student.")

class StudentAcademicProfile(BaseTool):
    name = "StudentAcademicProfile"
    description = """"
        Creates an academic profile of a student.
        This tool requires as input all the data about the student.
    """



    def _run(self, input: str) -> str:
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPEN_AI_KEY")
        )

        parser = JsonOutputParser(pydantic_object=AcademicProfile)

        template = PromptTemplate(
            template = """
                - Format the student data to his academic profile.
                - With the data, identify the universities suggested to him, and related with the interest of the student.
                - Focus on exaclty what make sense of informations expected by the universities.
                
                Current Information:
                {student_data}
                {output_format}
            """,
            input_variables=["student_data"],
            partial_variables={"output_format": parser.get_format_instructions()}
        )

        cadeia = template | llm | parser
        return cadeia.invoke({"student_data": input})
