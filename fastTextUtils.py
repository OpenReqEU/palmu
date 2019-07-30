import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import numpy as np 
import pandas as pd 
from gensim.models import FastText
import string


class FastTextUtils():


	def __init__(self , model_path ):


		self.model_path = model_path
		self.dim = 0 
		self.load_model()
		return None

	def load_model(self):

		self.model =  FastText.load_fasttext_format( self.model_path )

		v = self.model["example"]
		self.dim = v.shape[0]

		return True




