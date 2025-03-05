"""
Base class for benchmark use cases.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class BaseUseCase(ABC):
    """
    Abstract base class for benchmark use cases.
    All specific use cases should inherit from this class.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize a use case.
        
        Args:
            name: Name of the use case
            description: Description of the use case
        """
        self.name = name
        self.description = description
        self.inputs = {}
    
    def set_inputs(self, inputs: Dict[str, Any]) -> None:
        """
        Set inputs for the use case.
        
        Args:
            inputs: Dictionary of input parameters
        """
        self.inputs = inputs
    
    @abstractmethod
    def run_autogen(self, **kwargs) -> Dict[str, Any]:
        """
        Run the use case using AutoGen.
        
        Returns:
            Dictionary with results
        """
        pass
    
    @abstractmethod
    def run_semantic_kernel(self, **kwargs) -> Dict[str, Any]:
        """
        Run the use case using Semantic Kernel.
        
        Returns:
            Dictionary with results
        """
        pass
    
    @abstractmethod
    def run_langchain(self, **kwargs) -> Dict[str, Any]:
        """
        Run the use case using LangChain.
        
        Returns:
            Dictionary with results
        """
        pass
    
    @abstractmethod
    def run_crewai(self, **kwargs) -> Dict[str, Any]:
        """
        Run the use case using CrewAI.
        
        Returns:
            Dictionary with results
        """
        pass
    
    def get_framework_methods(self) -> Dict[str, callable]:
        """
        Get a mapping of framework names to their implementation methods.
        
        Returns:
            Dictionary mapping framework names to methods
        """
        return {
            "autogen": self.run_autogen,
            "semantic_kernel": self.run_semantic_kernel,
            "langchain": self.run_langchain,
            "crewai": self.run_crewai
        }
    
    def run(self, framework: str, **kwargs) -> Dict[str, Any]:
        """
        Run the use case with the specified framework.
        
        Args:
            framework: Name of the framework to use
            **kwargs: Additional arguments to pass to the framework method
            
        Returns:
            Dictionary with results
            
        Raises:
            ValueError: If the framework is not supported
        """
        framework = framework.lower()
        framework_methods = self.get_framework_methods()
        
        if framework not in framework_methods:
            raise ValueError(f"Unsupported framework: {framework}")
        
        return framework_methods[framework](**kwargs)
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the use case.
        
        Returns:
            Dictionary with use case metadata
        """
        return {
            "name": self.name,
            "description": self.description,
            "supported_frameworks": list(self.get_framework_methods().keys())
        } 