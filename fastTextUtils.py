

import numpy as np 
import pandas as pd 
from gensim.models import FastText
import string


class FasTextUtils():


	def __init__(self , model_path ):


		self.model_path = model_path
		self.dim = 0 
		self.load_model()
		return None

	def load_model(self):

		self.model =  FastText.load_fasttext_format( path )

		v = self.model["example"]
		self.dim = v.shape[0]

		return True

	def text_clean( self , text ):
		# 
		text = text.lower()
		text = text.encode("utf-8" , "ignore")
		text = text.translate( string.maketrans("",""), string.punctuation  )
		return text 


	def get_embedding( self , text ):

		text = self.text_clean( text )

		embedding = np.zeros(  ( self.dim ) )
		words = text.split(" ")
		for word in words:

			emb = self.model[word]
			embedding += emb 

		embedding /= len( words )

		return embedding 


path = "../fastText/qtmodel.bin"
ft = FasTextUtils( path )

emb = ft.get_embedding( "que es mi barco mi tesoro, mi pendon mi libertad")
print(emb)


