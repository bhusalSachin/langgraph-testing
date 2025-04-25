from ..utils import PromtTemplate, create_structured_llm, invoke_graph, obj_to_str
from ..models import OverallState, Node_description

from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import SystemMessage

node_description_promt = PromtTemplate(template="""
You are a workflow developer tasked with characterizating a graph. 
You have focused on LangChain and LangGraph frameworks in python.
Using the data below, describe what a node is for:

general graph description: {{graph_description}}

node name: {{node_name}}
type: {{type}}
{% if node_description %} previous description : {{node_description}} {% endif %}

income nodes: {{income_nodes}}
sample_input: {{input}}

outcome nodes: {{outcome_nodes}}
sample_output: {{output}}

{% if functions %}functions: {{functions}} {% endif %}

Take your time and be clrear.

First, identify the node name and its type.
Then look at the input_node, sample_input, and output_node, sample_output. 
Explain how it could interact with neighboring nodes.
Explain the input and output requirements.
{% if node_description %}Combine previous description and current description.{% endif %}
{% if functions %}figure out what the fuction are for in the graph context.{% endif %} 
Find out how the node can contribute to achieve the description. 
Finally, write the description of the node.""", 
input_variables=["graph_description", "input", "output", "node_name", "type", "functions", "income_nodes", "outcome_nodes", "node_description"])


# nodes
def generate_node_descriptions(state: OverallState):
    structured_llm = create_structured_llm(Node_description)

    config, error, error_message  = invoke_graph(graph=state["compiled_graph"],
                          input=state["valid_input"])
    
    if error:
        raise ValueError(f"Invalid graph input: {error}")
    
    configurable = {"configurable": config}

    history = list(state["compiled_graph"].get_state_history(configurable))
    history.reverse()

    node_name_in_tasks = [item.tasks[0].name for item in history if item.tasks]
    node_name_in_tasks.remove('__start__')
    
    node_tasks_in_tasks = [item.tasks[0].result for item in history if item.tasks]

    summary_graph = state["summary_graph"]

    for index, node_name in enumerate(node_name_in_tasks):
        current_description = summary_graph.nodes[node_name].get("description", None)
        functions = summary_graph.nodes[node_name].get("tools", None)

        actual_input = node_tasks_in_tasks[index]
        actual_output = node_tasks_in_tasks[index+1]

        parameters = {"graph_description":state["user_description"],
        "input":obj_to_str(actual_input),
        "output":obj_to_str(actual_output),
        "node_name":node_name,
        "type":str(summary_graph.nodes[node_name]["type"]),
        "functions":functions,
        "income_nodes":str(summary_graph.in_edges(node_name)),
        "outcome_nodes":str(summary_graph.out_edges(node_name)),
        "node_description":current_description}
                                                       
        system_message = node_description_promt.render(**parameters)
        llm_description = structured_llm.invoke([SystemMessage(system_message)])

        summary_graph.nodes[node_name]["description"] = llm_description.node_description

    
    return {"execution_configs": [config],
            "summary_graph": summary_graph}