from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from agent_openai_functions import AgentOpenAIFunctions

load_dotenv()


agent_openai = AgentOpenAIFunctions()
executor = AgentExecutor(agent=agent_openai.agent, tools=agent_openai.tools, verbose=True)

resp = executor.invoke({"input": "USP or Unicamp? which you recommend for the academic student Ana?"})
print(resp)