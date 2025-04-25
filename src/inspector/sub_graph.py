from langgraph.graph import END, START, StateGraph

from .models import SubGraphState
from .agents.generate_new_inputs import generate_new_inputs

sub_workflow = StateGraph(SubGraphState)

sub_workflow.add_node("generate_new_inputs", generate_new_inputs)

sub_workflow.set_entry_point("generate_new_inputs")
sub_workflow.set_finish_point("generate_new_inputs")