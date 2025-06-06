from langchain_core.messages import HumanMessage, SystemMessage

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.count.graph import workflow as testing_workflow
from src.inspector.graph import graph as inspector_workflow

user_description = ""

user_valid_input = {"messages": [HumanMessage(content="")]}

graph_before_compile = testing_workflow

configurations = config={
            "recursion_limit": 150,
            "thread_id": f"42" 
        }

result = inspector_workflow.invoke({"user_description":user_description
                    ,"valid_input": user_valid_input, 
                    "graph_before_compile": graph_before_compile},
                    config=configurations)