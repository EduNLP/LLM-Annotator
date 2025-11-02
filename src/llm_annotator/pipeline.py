import confection
import inspect

from llm_annotator.config import ModelType, ModelConfig
from llm_annotator.utils import valid_kwargs, components

from typing import Callable


class Pipeline:
    def __init__(self, config: dict[str, dict[str, str]]):
        self.components: list[Callable] = []
        self.config = confection.Config(config)

    def __call__(self, *args, **kwargs):
            outputs = {}
            # 1. CRITICAL FIX: Create a 'state' dictionary 
            #    This is a copy of the config and holds all component outputs
            state = dict(self.config) 
    
            for name, component in self.components:
                component_params = inspect.signature(component).parameters
                
                # 2. Draw arguments from the dynamic 'state' dictionary
                component_kwargs = {
                    param: state[param] 
                    for param in component_params 
                    if param in state
                }
    
                # Compute the outputs
                output_name, output = component(**component_kwargs)
                
                # 3. CRITICAL FIX: Update the 'state' dictionary 
                #    (which is temporary for this call), NOT self.config
                state[output_name] = output 
                
                outputs[output_name] = output
    
            # Handle single output (Bug #1 fix)
            if len(outputs) == 1:
                outputs = next(iter(outputs.values())) 
            return outputs

    def add_pipe(self, name: str, idx: int):
        component_factory = components.get(name)
        component = component_factory(**valid_kwargs(self.config, component_factory))

        new_element = (name, component)
        self.components.insert(idx, new_element)

