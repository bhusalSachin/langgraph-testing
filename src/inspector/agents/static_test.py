from ..models import OverallState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition

import networkx as nx

def static_test(state: OverallState):
    memory = MemorySaver() # it could be a SQLite database
    graph_after_compile = state["graph_before_compile"].compile(checkpointer=memory)

    graph_object = graph_after_compile.get_graph()

    nodes = graph_object.nodes
    edges = graph_object.edges

    graph_sumary = nx.DiGraph()

    for name, node in nodes.items():
        tools = {}
        type_node = type(node.data)

        if type_node == ToolNode:
            for name_tool, tool in node.data.tools_by_name.items():
                tools[name_tool] = tool.description

        graph_sumary.add_node(name, type=type_node, runnable=node.data, tools=tools, name=name) 

    for edge in edges:
        graph_sumary.add_edge(edge.source, edge.target, conditional=edge.conditional)

    return {"compiled_graph": graph_after_compile,
             "summary_graph": graph_sumary,
             "execution_configs": []}