import os

from langchain.tools import BaseTool
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

class EmailExtractor(BaseModel):
    text: str = Field("body of the email with the information about the grades adn review of the user student.")


class EmailData(BaseTool):
    name = "EmailData"
    description = ("""This tool is responsible for writing a email's body for a user student with a overview about his/her performance.
                    This tool requires as input all the information and history data of the user stundent.
                    You need to get the student data before invoke me. And the result of the student data should be passed as input to me.
                   """)

    def _run(self, user_student_history: str) -> str :
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPEN_AI_KEY")
        )

        parser = JsonOutputParser(pydantic_object=EmailExtractor)

        template = PromptTemplate(
            template="""You should analyze the {user_student_info_and_history} info received as input with all information and grades, and then create a email body well written with the overview to
            to the user student.
            
            Current Informations:
            {user_student_info_and_history}
            {output_format}
            """,
            input_variables=["user_student_info_and_history"],
            partial_variables={"output_format": parser.get_format_instructions()})

        cadeia = template | llm | parser

        email_content = cadeia.invoke({"user_student_info_and_history": input})
        return email_content