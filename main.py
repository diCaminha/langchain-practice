from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from agent_openai_functions import AgentOpenAIFunctions

load_dotenv()


agent_openai = AgentOpenAIFunctions()
executor = AgentExecutor(agent=agent_openai.agent, tools=agent_openai.tools)

resp = executor.invoke({"input": "what are the data about Ana?"})
print(resp)