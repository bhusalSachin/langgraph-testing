import copy
import uuid
from typing import (Any, Dict, List, Optional, Set, Tuple, Type,
                    Union)
from jinja2 import Template
from langchain_core.messages import (AIMessage, ChatMessage, FunctionMessage,
                                     HumanMessage, SystemMessage, ToolMessage)
from langgraph.graph.graph import CompiledGraph
from pydantic import BaseModel
from typing_extensions import TypedDict

from src.config import llm

def create_structured_llm(structured_output: BaseModel):
    """
    Creates a structured language model (LLM) based on the provided configuration and structured output model.
    Args:
        config (dict): A dictionary containing the configuration for the LLM. It should have a key "configurable" 
                       which contains another dictionary with the key "llm" representing the language model.
        structured_output (BaseModel): An instance of a BaseModel that defines the structure of the output.
    Returns:
        The structured language model configured with the provided structured output.
    Raises:
        ValueError: If the LLM model is not valid or cannot be configured with the structured output.
    """

    try:
            model = llm
            structured_llm = model.with_structured_output(structured_output)
            return structured_llm
    except:
        raise ValueError("The llm model is not valid")
    
class PromtTemplate:
    """A class to render a prompt template with input variables."""
    def __init__(self, template: str, input_variables: list[str]):
        self.template = Template(template)
        self.input_variables = input_variables

    def render(self, **kwargs):
        return self.template.render(**kwargs)
    
class Config(TypedDict):
    user_id: uuid.uuid4
    thread_id: uuid.uuid4
    description: str

def invoke_graph(graph: CompiledGraph, 
                 input: Any,
                 thread_id:Optional[str] = None,
                 user_id:Optional[str]= None,
                 description:str="") -> tuple[Config, bool, str]:
    

    thread_id = thread_id if thread_id else str(uuid.uuid4())
    user_id = user_id if user_id else str(uuid.uuid4())
    
    config = Config(thread_id=thread_id,
                    user_id=user_id,
                    description=description)
    
    configurable = {"configurable": config}

    error = False
    error_message = ""

    try:
        graph.invoke(input, config=configurable, stream_mode="debug")
    except Exception as e:
        error_message = f"Graph execution failed: {e}"
        error = True

    return config, error, error_message

def reduce_valid_input(left: Any | None, right: Any | None) -> Any:
    if left is None:
        return right
    if right is None:
        return left
    if left == right:
        return left
    if left != right:
        return left

def generate_pairs(a: list, b: list) -> list[tuple]:
    """
    Generate all possible pairs of elements from two lists.
    Args:
        a (list): The first list of elements.
        b (list): The second list of elements.
    Returns:
        list[tuple]: A list of tuples, where each tuple contains one element from list 'a' and one element from list 'b'.
    """

    result = []
    for node in a:
        for tester in b:
            result.append((node, tester))

    return result

class TypeAnnotator:
    _iterables = [list, tuple, set, dict]
    _message_types = [HumanMessage, AIMessage, ToolMessage, SystemMessage, 
                     FunctionMessage, ChatMessage]
    _no_iterables = [int, float, str, bool] + _message_types

    def __init__(self, obj: Any):
        self.obj = obj

    def get_type(self) -> Type:
        """Get the type annotation directly as a typing object."""
        return self._infer_type(self.obj)

    def _infer_type(self, obj: Any) -> Type:
        """Recursively determine the type annotation of a complex structure."""
        # Handle message types first
        if any(isinstance(obj, t) for t in self._message_types):
            return type(obj)
        
        # Handle basic types
        if type(obj) in self._no_iterables:
            return type(obj)

        # Handle collections
        handlers = {
            list: self._handle_list,
            dict: self._handle_dict,
            tuple: self._handle_tuple,
            set: self._handle_set
        }
        return handlers.get(type(obj), lambda x: type(x))(obj)

    def _handle_list(self, obj: List) -> Type[List]:
        """Handle list type annotation."""
        if not obj:
            return List[Any]
        
        types = {self._infer_type(el) for el in obj}
        if len(types) == 1:
            return List[next(iter(types))]
        return List[Union[tuple(sorted(types, key=str))]]

    def _handle_dict(self, obj: Dict) -> Type[Dict]:
        """Handle dict type annotation."""
        if not obj:
            return Dict[Any, Any]
        
        key_types = {self._infer_type(k) for k in obj.keys()}
        value_types = {self._infer_type(v) for v in obj.values()}
        
        key_type = (Union[tuple(sorted(key_types, key=str))] 
                   if len(key_types) > 1 else next(iter(key_types)))
        value_type = (Union[tuple(sorted(value_types, key=str))] 
                     if len(value_types) > 1 else next(iter(value_types)))
        
        return Dict[key_type, value_type]

    def _handle_tuple(self, obj: Tuple) -> Type[Tuple]:
        """Handle tuple type annotation."""
        if not obj:
            return Tuple[()]
        return Tuple[tuple(self._infer_type(el) for el in obj)]

    def _handle_set(self, obj: Set) -> Type[Set]:
        """Handle set type annotation."""
        if not obj:
            return Set[Any]
        
        types = {self._infer_type(el) for el in obj}
        if len(types) == 1:
            return Set[next(iter(types))]
        return Set[Union[tuple(sorted(types, key=str))]]
    
def obj_to_str(obj, max_depth=float('inf'), current_depth=0):
    """
    Converts any Python object into a string representation that looks like the original code.
    
    Args:
        obj: Any Python object
        max_depth: Maximum depth for recursion (default: infinite)
        current_depth: Current recursion depth (used internally)
        
    Returns:
        String representation of the object that looks like code
    """
    # Check if we've reached maximum depth
    if current_depth >= max_depth:
        return repr(obj)
    
    if isinstance(obj, dict):
        items = [f'"{k}": {obj_to_str(v, max_depth, current_depth + 1)}' for k, v in obj.items()]
        return '{' + ', '.join(items) + '}'
    elif isinstance(obj, (list, tuple)):
        items = [obj_to_str(item, max_depth, current_depth + 1) for item in obj]
        return '[' + ', '.join(items) + ']' if isinstance(obj, list) else '(' + ', '.join(items) + ')'
    elif isinstance(obj, str):
        return f'"{obj}"'
    elif isinstance(obj, (int, float, bool, type(None))):
        return str(obj)
    elif obj.__class__.__module__ == 'builtins':
        return repr(obj)
    else:
        # Handle custom objects by reconstructing their initialization
        class_name = obj.__class__.__name__
        
        # If at max_depth, just return the repr
        if current_depth >= max_depth:
            return f"{class_name}(...)"
        
        # Try to get the object's attributes
        try:
            # First try to get __dict__
            attrs = copy.copy(obj.__dict__)

            # more clear messages representation
            attrs.pop('additional_kwargs', None)
            attrs.pop('usage_metadata', None)
            attrs.pop('response_metadata', None)
            
        except AttributeError:
            try:
                # If no __dict__, try getting slots
                attrs = {slot: getattr(obj, slot) for slot in obj.__slots__}
            except AttributeError:
                # If neither works, just use repr
                return repr(obj)
        
        # Convert attributes to key=value pairs
        attr_strs = []
        for key, value in attrs.items():
            # Skip private attributes (starting with _)
            if not key.startswith('_'):
                attr_strs.append(f"{key}={obj_to_str(value, max_depth, current_depth + 1)}")
        
        return f"{class_name}({', '.join(attr_strs)})"
