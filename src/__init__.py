"""
Advanced Model Compression Pipeline for IoT Deployment
"""

__version__ = "1.0.0"
__author__ = "Abdullah Aldwsry, Wahaq Almutairi, Meshari Alshammari, Hossam Baroudi, Mazen Hamze"

from . import config
from . import models
from . import evaluation
from . import utils
from . import data_loader

__all__ = [
    'config',
    'models', 
    'evaluation',
    'utils',
    'data_loader'
]
