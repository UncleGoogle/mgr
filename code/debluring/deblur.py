import sys
import os.path
sys.path.insert(1, os.path.join(os.path.dirname(__file__), '..'))

from helpers import readImages


data_path = os.path.join(os.path.dirname(__file__), 'sample_datla')

# if not os.path.exists(data_path):
#     raise Ini

dot, img = readImages([data_path])
