from state import AgentState
from tools import search_documents

def tool_node(state: AgentState):
    last_message = state["messages"][-1]
    query = last_message.content

    result = search_documents(query)

    return {
        "messages": state["messages"] + [
            {"role": "tool", "content": result}
        ]
    }