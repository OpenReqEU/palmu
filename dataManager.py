
import pandas as pd 
import numpy as np
import json
import os
import pickle 
import tables
import faiss 

class DataManager():


	def __init__(self , gloveFile = "./path" , emb_dim = 200 ):

		# create model. 

		self.model_glove = self.loadGloveModel( gloveFile )
		self.emb_dim = emb_dim
		# create files 
		self.process_files( self.model_glove )

		return None

	def buildIndex( self ):


	def reqInData( self , idd ):

		return idd in self.mapping.keys()

	def loadGloveModel( self , gloveFile):

		print("Loading Glove Model")
		f = open(gloveFile,'r')
		model = {}
		for line in f:
			splitLine = line.split()
			word = splitLine[0]
			embedding = np.array([float(val) for val in splitLine[1:]])
			model[word] = embedding
		print("Done.",len(model)," words loaded!")
		return model

	def process_files( self , model_glove ):
	    # this function saves on disk the mappings in between the vector embeddings and the 
	    # List existing files on data folder , 
		files = os.listdir( "./data/")
		files_json = [ "./data/"+f for f in files if ".json" in f ]
		embs , mapp = self.get_embeddings( files_json   )
		self.mapping = mapp 
		embs = np.array( embs )

		pickle.dump( mapp ,   open( "./data/mappings200.map", "wb" ) , protocol=2 )
		np.save( "./data/embbedings200.npy" , embs   )

		return True

	def get_embeddings( self ,  files_json  ):
		# return the 
		# id - > embeddings correspondence 
		all_embeddings = []
		comment_embeddings = [] 
		index = 0
		mapping = {}
		print( files_json )
		for file in files_json:
			print( file )
			requirements = self.get_reqs( file )

			for req in requirements:

				if "text" in req.keys():
				#print( req["text"])
					name_emb = self.get_embedding_txt( req["name"] , self.model_glove )
					embedding = self.get_embedding_txt( req["text"]  , self.model_glove )
					comment_emb = self.get_embedding_com( req , self.model_glove )
					component_emb = self.get_embedding_components( req , self.model_glove )

					embedding =  0.1*name_emb + 0.5*embedding + 0.3*comment_emb + 0.1*component_emb 
					#embedding is a vector 
					mapping[req["id"]] = index
					index  = index + 1
					all_embeddings.append( embedding )

		return all_embeddings , mapping 

	def get_reqs( self , file ):
	    
		data = ""
		with open( file , "r") as f:
			data = f.read()

			data = json.loads( data )
			print(" getting requirements from {}  - number of reqs: {}".format(  file , len(data["requirements"])) )
		return data["requirements"]

	def get_embedding_txt(self ,  txt  , model ):

		if txt is None or txt is "" or txt is " ":
			return 0
		#txt = req["text"]
		# here return the embedding
		txt = txt.lower() 
		words = txt.split(" ")
		embds = np.zeros( ( self.emb_dim))
		for w in words:
			emb = None 
			try: 
			#if w in model.keys():
			    emb = model[w]
			except:
			    emb = np.zeros( (self.emb_dim) )
		    
			embds += emb

		embds = embds / len( words )
		return embds

	def get_embedding_components( self ,  req , model ):
		if "requirementParts" not in req or -7 not in req["requirementParts"]:
			return 0
		components_dict = req["requirementParts"][-1]
		if "text" not in components_dict.keys():
			return np.zeros( (self.emb_dim))
		text_list = components_dict["text"] #.replace( '"' , "")[1:-1] #.split(",") #  .split('"' )
		embs = np.zeros( (self.emb_dim))
		for word in text_list:
			if word in model.keys():
	 			emb = model[word]
				embs += emb 
		embs = embs / (len( text_list ) + 1 )
		return embs 
	        

	def get_embedding_com( self , req , model ):
		if "comments" not in req:
			return 0
		embs = np.zeros( ( self.emb_dim ))
		for comment in req["comments"]:
			txt = comment["text"]
			emb = self.get_embedding_txt( txt , model  )
			embs += emb 
		embs = embs/(len( req["comments"]  ) + 1 )
		return embs 
