from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .models import GraphState

workflow = StateGraph(GraphState)

from .agent import count, stop_count

workflow.add_node("keep_counting", count)

workflow.add_edge(START, "keep_counting")
workflow.add_conditional_edges(
    "keep_counting",
    stop_count,
    {
        "True": END,
        "False": "keep_counting"
    }
)

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

def get_graph_app():
    return app
