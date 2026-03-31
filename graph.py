'''The orchestrator for the agent's workflow.'''
from langgraph.graph import StateGraph
from state import AgentState
from nodes.agent_node import agent_node
from nodes.tool_node import tool_node

def should_use_tool(state: AgentState):
    last_message = state["messages"][-1].content.lower()

    if "document" in last_message:
        return "tool"
    return "end"

workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tool", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_use_tool,
    {
        "tool": "tool",
        "end": "__end__"
    }
)

workflow.add_edge("tool", "agent")

app = workflow.compile()