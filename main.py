from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from agent_openai_functions import AgentOpenAIFunctions

load_dotenv()


agent_openai = AgentOpenAIFunctions()
executor = AgentExecutor(agent=agent_openai.agent, tools=agent_openai.tools, verbose=True)

resp = executor.invoke({"input": "Is Marcos going well in math classes?"})
print(resp)