from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .models import OverallState

from .agents.static_test import static_test
from .agents.generate_test_cases import generate_test_cases, more_test_cases
from .agents.generate_testers import generate_testers
from .agents.node_description import generate_node_descriptions
from .agents.analyze import analize_results, more_results

from .sub_graph import sub_workflow as sub_builder

workflow = StateGraph(OverallState)

workflow.add_node("static_test", static_test)
workflow.add_node("generate_node_descriptions", generate_node_descriptions)
workflow.add_node("generate_testers", generate_testers)
workflow.add_node("generate_test_cases", generate_test_cases)
workflow.add_node("run_test_cases", sub_builder.compile())
workflow.add_node("analize_results", analize_results)


workflow.set_entry_point("static_test")
workflow.add_edge("static_test", "generate_node_descriptions")
workflow.add_edge("generate_node_descriptions", "generate_testers")
workflow.add_edge("generate_testers", "generate_test_cases")
workflow.add_conditional_edges("generate_test_cases", more_test_cases, ["generate_test_cases", "run_test_cases"])
workflow.add_edge("run_test_cases", "analize_results")
workflow.add_conditional_edges("analize_results", more_results, {True: "analize_results", False: "__end__"})

graph = workflow.compile()