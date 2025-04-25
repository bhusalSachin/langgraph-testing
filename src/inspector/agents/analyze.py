from ..utils import PromtTemplate, create_structured_llm, invoke_graph, obj_to_str
from ..models import OverallState, FinalOutput

from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage

# promts
assertion_prompt = PromtTemplate("""
{{role_description}}
A test cases has been run on the graph and here you have the results. 
You must validate the results using the test case description, acceptance criteria, and the output of the test case: 
                                 
- test case name: {{test_case_name}}
- test case description: {{test_case_description}}
- acceptance criteria: {{acceptance_criteria}}
- output: {{output}}
                                 
You must validate the output. If the output is as described in the acceptance criteria, return 'True'. Otherwise, return 'False'.
Finally, write additional comments of how to solve the issue if the output is not as expected.
If the output is as expected, the comments should be a description of the behavior of the graph.
""", input_variables=["test_case_name" ,"role_description", "test_case_description", "acceptance_criteria", "output"])

# Nodes
def analize_results(state: OverallState):
    structured_llm = create_structured_llm(FinalOutput)

    current_result_config = state["execution_configs"].pop(0)

    if not current_result_config["description"]:
        return {"listResults": []}
    
    for test_case in state["test_cases"]:
        if test_case.id == current_result_config["thread_id"]:
            current_test_case = test_case
            break  

    tester = state["testers"][current_result_config["user_id"]] 

    configurable = {"configurable": current_result_config}

    parameters = {"test_case_name" : current_test_case.name,
                  "role_description":tester.description,
                  "test_case_description":current_test_case.description,
                  "acceptance_criteria":current_test_case.acceptance_criteria,
                  "output":obj_to_str(state["compiled_graph"].get_state(configurable).values)}

    system_message = assertion_prompt.render(**parameters)
    final_output = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Generate the final output.")])
    final_output.tester_id = tester.id
    final_output.test_case_id = current_test_case.id

    return {"listResults": [final_output]}

# conditional edges
def more_results(state: OverallState):
    return bool(state["execution_configs"])