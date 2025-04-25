import gradio as gr

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.test import result
from src.inspector.utils import obj_to_str

CHECKMARK = "\u2705"  # ✅
CROSS = "\u274C"      # ❌

with gr.Blocks() as demo:
    for result_graph in result["listResults"]:
        symbol = CHECKMARK if result_graph.assertion else CROSS

        for test_case in result["test_cases"]:
            if test_case.id == result_graph.test_case_id:
                current_test_case = test_case
                break

        tester = result["testers"][result_graph.tester_id] 

        configurations = {"configurable": {"user_id":tester.id, 
                                           "thread_id":current_test_case.id}}

        with gr.Accordion(f"{current_test_case.name}: {symbol}", open=False):
            gr.Markdown(f"{result_graph.comments}")

            with gr.Accordion(f"Details", open=False):
                gr.Markdown(f"Tester: {tester.role}")
                gr.Markdown(f"Teste description: {current_test_case.description}")
                gr.Markdown(f"Teste assertion: {current_test_case.acceptance_criteria}")
                gr.Markdown(f"Actual output: {obj_to_str(result["compiled_graph"].get_state(configurations).values)}")

demo.launch(debug=False, inbrowser=False)