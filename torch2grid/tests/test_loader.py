import unittest
import tempfile
import os
import sys
import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch2grid.loader import load_torch_model



class TestLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil