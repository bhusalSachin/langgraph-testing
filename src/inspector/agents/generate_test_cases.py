from ..utils import PromtTemplate, create_structured_llm, invoke_graph, obj_to_str
from ..models import OverallState, TaseCasesList

from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.constants import Send

# promts
test_case_prompt = PromtTemplate("""
{{role_description}}

You must test this node deeply. The below is the node information:
                            
name: {{node_name}}
type: {{node_type}}
description: {{node_description}}
functions: {{node_functions}}
sample_input: {{sample_input}}
sample_output: {{sample_output}}
                               
existing test cases: {{existing_test_cases}}
                            
How would you test the node? 
Give at least 3 test case.
AVOID [repeating the same test case, puting values in the acceptance_criteria]                      
Take your time and think out of the box.
If there is no test case neded, return and empty object.""",
input_variables=["role_description", "node_name", "node_type", "node_description", "node_functions", "sample_input", "sample_output", "existing_test_cases"])

# Nodes
def generate_test_cases(state: OverallState):
    structured_llm = create_structured_llm(TaseCasesList)

    current_node_and_tester = state["node_and_tester"].pop(0)
    current_node = current_node_and_tester[0]
    current_tester = current_node_and_tester[1]

    configuration = state["execution_configs"][0]
    configurable = {"configurable": configuration}

    history = list(state["compiled_graph"].get_state_history(configurable))
    history.reverse()

    node_tasks_in_tasks = [(item.tasks[0].name, item.tasks[0].result) for item in history if item.tasks]

    actual_inputs = []
    actual_outputs = []

    for index, task in enumerate(node_tasks_in_tasks):
        if task[0] == current_node["name"]:
            actual_inputs.append(node_tasks_in_tasks[index-1][1])
            actual_outputs.append(task[1])

    name_test_cases = [test_case.name for test_case in state["test_cases"]]
    
    parameters = {"role_description":current_tester.description,
                  "node_name":current_node["name"],
                  "node_type":current_node["type"],
                  "node_description":current_node["description"],
                  "node_functions":current_node["tools"],
                  "sample_input":obj_to_str(actual_inputs),
                  "sample_output":obj_to_str(actual_outputs),
                  "existing_test_cases":name_test_cases}
    
    system_message = test_case_prompt.render(**parameters)
    test_cases = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Generate the set of test cases.")])

    for test_case in test_cases.test_cases:
        test_case.tester_id = current_tester.id

    return {"test_cases": test_cases.test_cases}

# conditional edges
def more_test_cases(state: OverallState):
    if state["node_and_tester"]:
        return "generate_test_cases"
    else:
        routing = []
        valid_inpout = state["valid_input"]
        compiled_graph = state["compiled_graph"]
        execution_configs = state["execution_configs"]

        for test_case in state["test_cases"]:
            new_state = {"current_test_case":test_case, 
                         "valid_input":valid_inpout,
                         "compiled_graph":compiled_graph}
                    
            routing.append(Send("run_test_cases",new_state))

        return routing