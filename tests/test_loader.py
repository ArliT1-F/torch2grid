"""
Tests for torch2grid loader functionality.
"""

import unittest
import tempfile
import os
import sys
import torch

# Add the parent directory to the path so we can import torch2grid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch2grid.loader import load_torch_model


class TestLoader(unittest.TestCase):
    """Test model loading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_state_dict(self):
        """Test loading a state_dict."""
        # Create a simple state_dict
        state_dict = {
            'layer1.weight': torch.randn(10, 5),
            'layer1.bias': torch.randn(10),
        }
        
        # Save to file
        model_path = os.path.join(self.temp_dir, 'test_model.pt')
        torch.save(state_dict, model_path)
        
        # Load it back
        loaded = load_torch_model(model_path)
        
        # Check that it's the same
        self.assertEqual(loaded.keys(), state_dict.keys())
        for key in state_dict:
            self.assertTrue(torch.equal(loaded[key], state_dict[key]))
    
    def test_load_model(self):
        """Test loading a complete model."""
        # Create a simple model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        )
        
        # Save to file
        model_path = os.path.join(self.temp_dir, 'test_model.pth')
        torch.save(model, model_path)
        
        # Load it back
        loaded = load_torch_model(model_path)
        
        # Check that it's a model
        self.assertIsInstance(loaded, torch.nn.Module)
    
    def test_file_not_found(self):
        """Test handling of non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_torch_model('nonexistent_file.pt')
    
    def test_invalid_file(self):
        """Test handling of invalid file."""
        # Create a text file (not a valid PyTorch file)
        invalid_path = os.path.join(self.temp_dir, 'invalid.txt')
        with open(invalid_path, 'w') as f:
            f.write("This is not a PyTorch file")
        
        with self.assertRaises(RuntimeError):
            load_torch_model(invalid_path)
    
    def test_directory_path(self):
        """Test handling of directory path instead of file."""
        with self.assertRaises(FileNotFoundError):
            load_torch_model(self.temp_dir)


if __name__ == '__main__':
    unittest.main()