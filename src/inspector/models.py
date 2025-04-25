from pydantic import BaseModel, Field, PrivateAttr
from typing import (Annotated, Any, Dict, List, Optional, Set, Tuple, Type,
                    Union)
from typing_extensions import TypedDict
import uuid
import operator

from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph

import networkx as nx
import gradio as gr

from .utils import reduce_valid_input, Config

class Node_description(BaseModel):
    node_description: str = Field(description="Description of the node. Max 45 words")

class SuggestedTester(BaseModel):
    role: str = Field(
        description="Role of the tester in the context of the graph.",
    )
    description: str = Field(
        description="Role description of the tester expertise, focus, concerns, and motives. (you are ...) ",
    )
    _id: str = PrivateAttr(default_factory=lambda: str(uuid.uuid4()))

    @property
    def id(self):
        return self._id
    
class Testers(BaseModel):
    testers: List[SuggestedTester] = Field(
        description="Comprehensive list of testers with their roles and descriptions",
    )

class TestCase(BaseModel):
    name: str = Field(description="name of the test case.")
    
    description: str = Field(description="Test case description")
    
    acceptance_criteria: str = Field(description="criteal to pass the test")
    
    tester_id: Optional[str] = Field(description="leave this field blank")

    _id: str = PrivateAttr(default_factory=lambda: str(uuid.uuid4()))

    @property
    def id(self):
        return self._id

class TaseCasesList(BaseModel):
    test_cases: List[TestCase] = Field(description="Comprehensive list of test cases with their properties")

class Input(BaseModel):
    new_input: str = Field(description="new input for the test case")
    tester_id: Optional[str] = Field(description="leave this field blank")
    test_case_id: Optional[str] = Field(description="leave this field blank")
    actual_input: Optional[
        Union[
            str, 
            Dict[str, Union[str, int, float, bool]], 
            List[Union[str, int, float, bool]]
        ]
    ] = Field(
        description="leave this field blank"
    )
    is_successful: Optional[bool] = Field(description="leave this field blank")

class FinalOutput(BaseModel):
    assertion : bool = Field(description="Assertion result of the test case")
    comments : str = Field(description="Comments on the test case output")
    tester_id: Optional[str] = Field(description="leave this field blank")
    test_case_id: Optional[str] = Field(description="leave this field blank")

class OverallState(TypedDict):
    # user input
    user_description: str
    valid_input: Annotated[Any, reduce_valid_input]
    graph_before_compile: StateGraph

    # internal use
    compiled_graph: Annotated[CompiledGraph, reduce_valid_input]
    summary_graph: nx.DiGraph
    execution_configs: Annotated[list[Config], operator.add]
    testers: dict[str, SuggestedTester]
    node_and_tester: list[tuple]
    test_cases: Annotated[list[TestCase], operator.add]
    all_new_inputs: Annotated[list[Input], operator.add]
    listResults: Annotated[list[FinalOutput], operator.add]

class SubGraphState(TypedDict):
    current_test_case: TestCase
    valid_input:  Annotated[Any, reduce_valid_input]
    all_new_inputs: Annotated[list[Input], operator.add]
    compiled_graph: Annotated[CompiledGraph, reduce_valid_input]
    execution_configs: Annotated[list[Config], operator.add]
