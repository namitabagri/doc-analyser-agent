from langchain_openai import ChatOpenAI
from state import AgentState

llm = ChatOpenAI(model="gpt-4o-mini")

def agent_node(state: AgentState):
    response = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}