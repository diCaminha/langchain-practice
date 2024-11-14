import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import create_openai_tools_agent
from langchain_community.chat_models import ChatOpenAI
from langchain_core.tools import Tool

from students_util import StudentData

load_dotenv()

class AgentOpenAIFunctions:

    def __init__(self):
        llm = ChatOpenAI(
            model="gpt-4o",
            api_key=os.getenv("OPEN_AI_KEY")
        )

        student_data_tool = StudentData()

        self.tools = [
            Tool(
                name=student_data_tool.name,
                description=student_data_tool.description,
                func=student_data_tool.run
            )
        ]

        prompt = hub.pull("hwchase17/openai-functions-agent")
        self.agent = create_openai_tools_agent(llm, self.tools, prompt)
