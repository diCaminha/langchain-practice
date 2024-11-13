import os
from tempfile import template

from dotenv import load_dotenv
from langchain.tools import BaseTool
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from openai import models
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

class StudentExtractor(BaseModel):
    name: str = Field("student first name, always in lowercase.")


class StudentData(BaseTool):
    name = "StudentData"
    description = "This tool is responsable for extract informations and history data of a student."

    def _run(self, input: str) -> str :
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPEN_AI_KEY")
        )

        parser = JsonOutputParser(pydantic_object=StudentExtractor)

        template = PromptTemplate(
            template="""You should analyse the {input} info and extract the name informed.
            Output format:
            {output_format}
            """,
            input_variables=["input"],
            partial_variables={"output_format": parser.get_format_instructions()})

        cadeia = template | llm | parser

        resp = cadeia.invoke({"input": input})
        print(resp)


question = "How Ana is going on math?"
StudentData().run({"input": question})
