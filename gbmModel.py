

import lightgbm as lgb 
import os
import numpy as np 
import pickle


class GBMModel( ):


	def __init__( self  , path = "" , name = "" ):

		# thic class aims to load gbm models 
		self.path = path
		self.name = name 
		self.models = []
		print("Loading LGB models")
		self.load_models()
		#print( len( self.models ) )

		return 


	def load_models( self ):

		files = os.listdir( self.path )

		files = [ x for x in files if self.name in x ]

		for f in files:
			path = self.path + "/" + f 
			l = lgb.Booster( model_file = path  )
			self.models.append( l )

		with open( self.path + "/logistic.pkl" , 'rb') as fid:
			self.logistic = pickle.load(fid)

		return True

	def get_top_k( self , data  ,  k  ):

		# data is an array -> [   word_emb_issue , word_emb_candidate_ ]
		# return the index of the highest ranked scores 

		#print( data.shape )
		


		y_preds  = np.zeros( ( len( data )  ))

		for  i , gbm in  enumerate( self.models ) :

			y = gbm.predict( data )
			y_preds += y



		y_preds = y_preds/len( self.models) 

		#iiix = y_preds.argsort()[-20:][::-1]
		#iiix = np.arange( len( y_preds ))

		#return iiix , y_preds[iiix ] 
		y_preds  = self.logistic.predict_proba( y_preds.reshape(  (-1 , 1 ))  )[: , 1 ]
		#clf.predict_proba( calibration_data)[:,1]
		indexs = y_preds.argsort()[-k:][::-1]
		#indexs =   np.where( y_preds == 1.0 )[0]   #y_preds.argsort()[-k:][::-1]
		#print( len( indexs )  )
		#print( len( indexs ) )
		#indexs =   y_preds.argsort()[-20:][::-1]
		#print( "valids:" , indexs )
		#print("maxxxx" ,  y_preds[ indexs ].max( ))
		return indexs  , y_preds[ indexs ]






