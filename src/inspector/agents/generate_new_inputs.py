from ..utils import PromtTemplate, create_structured_llm, invoke_graph, obj_to_str, TypeAnnotator
from ..models import Input,SubGraphState

from langchain_core.runnables.config import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage


new_input_prompt = PromtTemplate("""
You are a LangChain and LangGraph python developer. Your are focused on testing a graph of LangGraph.
Some senior testers have provided you with a test case for the graph.
The test case is as follows:
                                 
- name: {{test_case_name}}
- description: {{test_case_description}}
- graph valid input: {{graph_valid_input}}
                                 
you must follow this instructions:
1. Review the test case description.
2. Validate if the test case can be tested with an input using the valid input structure.
3. If it can't be tested, return an empty string.
4. If it can be tested, create a new imput for the test case.
5. verify carefully the new input format. Every open bracket must have a closing bracket and so on.
6. For each property in the input, you MUST make sure it is the same type as it is in valid input.
7. For any message object, the content must be a string.
8. Make sure the string could be passed to the 'eval' python function. For example, if the input has 'null' it should be 'None'.
9. Return the new input.""",
input_variables=["test_case_name", "test_case_description", "graph_valid_input"])

# Nodes
def generate_new_inputs(state: SubGraphState):
    print("\n")
    print("Producing new inputs for testing..")
    structured_llm = create_structured_llm(Input)

    parameters = {"test_case_name": state["current_test_case"].name,
                  "test_case_description": state["current_test_case"].description,
                  "graph_valid_input": obj_to_str(state["valid_input"])}

    system_message = new_input_prompt.render(**parameters)

    new_input = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Generate the new input.")])
    new_input.tester_id = state["current_test_case"].tester_id
    new_input.test_case_id = state["current_test_case"].id

    try:
        agent_valid_input = eval(new_input.new_input)
        agent_valid_input_type = TypeAnnotator(agent_valid_input).get_type()
        valid_input_type = TypeAnnotator(state["valid_input"]).get_type()

        if agent_valid_input_type == valid_input_type:
            new_input.actual_input = agent_valid_input

            config, error, error_message  = invoke_graph(graph=state["compiled_graph"],
                                                         input=agent_valid_input, 
                                                         description= state["current_test_case"].name,
                                                         thread_id=new_input.test_case_id,
                                                         user_id=new_input.tester_id)
            new_input.is_successful = not error

            print("New inputs been generated successfully.")
            
            return {"all_new_inputs": [new_input], 
                    "execution_configs": [config]}
        else:
            print("New inputs couldn't be generated. FAILED.")
            raise ValueError(f"invalid input type for {new_input.new_input}")

    except Exception as e:
        return {"all_new_inputs": []}