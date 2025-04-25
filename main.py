from langchain_core.messages import HumanMessage, SystemMessage

from src.count.graph import workflow as testing_workflow
from src.inspector.graph import graph as inspector_workflow

from src.inspector.models import FinalOutput

user_description = ""

user_valid_input = {"messages": [HumanMessage(content="")]}

graph_before_compile = testing_workflow

def print_test_results(results: FinalOutput):
    passed = []
    failed = []

    for result in results["listResults"]:
        test_case_id = result.test_case_id
        test_case = next(
                (
                    tc for tc in results["test_cases"]
                    if tc.tester_id == result.tester_id
                    and getattr(tc, "name", None)
                    and getattr(tc, "id", None) == result.test_case_id
                ),
                None
            )
        test_name = test_case.name if test_case else f"TestCase ID: {test_case_id}"

        if result.assertion:
            passed.append((test_name, result.comments))
        else:
            failed.append((test_name, result.comments))

    print("\n✅ Successful Test Cases\n")
    for name, comment in passed:
        print(f"✔ [{name}]\n   - Comments: {comment}\n")

    print("\n❌ Failed Test Cases\n")
    for name, comment in failed:
        print(f"✘ [{name}]\n   - Comments: {comment}\n")

if __name__ == "__main__":
    configurations = config={
            "recursion_limit": 150,
            "thread_id": f"42" 
        }

    result = inspector_workflow.invoke({"user_description":user_description
                    ,"valid_input": user_valid_input, 
                    "graph_before_compile": graph_before_compile},
                    config=configurations)

    print("\n\n")    
    print("-------RESULT---------")
    print("\n")
    print(print_test_results(result))
    
