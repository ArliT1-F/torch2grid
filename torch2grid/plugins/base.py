"""
Base class for transformer plugins.
"""

from abc import ABC, abstractmethod
import numpy as np


class TransformerPlugin(ABC):
    """
    Base class for custom transformer plugins.
    
    Plugins define how to transform a dictionary of tensors into a 2D grid
    for visualization. Subclass this and implement the transform() method.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique name for this plugin.
        
        Returns:
            String identifier for the plugin
        """
        pass
    
    @property
    def description(self) -> str:
        """
        Description of what this transformer does.
        
        Returns:
            Human-readable description
        """
        return "Custom transformer plugin"
    
    @abstractmethod
    def transform(self, tensors: dict) -> np.ndarray:
        """
        Transform tensors into a 2D grid.
        
        Args:
            tensors: Dictionary mapping layer names to numpy arrays
            
        Returns:
            2D numpy array representing the grid
        """
        pass
    
    def can_handle(self, tensors: dict) -> bool:
        """
        Check if this plugin can handle the given tensors.
        
        Override this method to add conditions for when your plugin
        should be used (e.g., only for specific layer types).
        
        Args:
            tensors: Dictionary mapping layer names to numpy arrays
            
        Returns:
            True if this plugin can handle these tensors
        """
        return True
    
    def preprocess(self, tensors: dict) -> dict:
        """
        Preprocess tensors before transformation.
        
        Override this to filter, normalize, or modify tensors.
        
        Args:
            tensors: Dictionary mapping layer names to numpy arrays
            
        Returns:
            Processed tensors dictionary
        """
        return tensors
    
    def postprocess(self, grid: np.ndarray) -> np.ndarray:
        """
        Postprocess the grid after transformation.
        
        Override this to apply additional transformations to the grid.
        
        Args:
            grid: 2D numpy array
            
        Returns:
            Processed grid
        """
        return grid
    
    def __call__(self, tensors: dict) -> np.ndarray:
        """
        Full transformation pipeline.
        
        Args:
            tensors: Dictionary mapping layer names to numpy arrays
            
        Returns:
            2D numpy array representing the grid
        """
        tensors = self.preprocess(tensors)
        grid = self.transform(tensors)
        grid = self.postprocess(grid)
        return grid
