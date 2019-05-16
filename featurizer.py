import pandas as pd 
import numpy as np
import json
import os
import pickle 
import tables
import faiss 



class Featurizer( ):


	def __init__( self , words_model , dim , encoder_reqstatus , encoder_reqtype   ):

		# this featurizer recieves an word2vec model as a dict 
		# and  will have methods to build the representation for a requirement. 
		# words
		# words_model ->  dict
		# dim -> int
		# label_encoder -> 
		self.words_model = words_model
		self.encoder_status = encoder_reqstatus
		self.encoder_type = encoder_reqtype
		self.dim = dim # model dimensionclasses_
		self.final_size = self.dim  #+ len(self.encoder_status.classes_) + len( self.encoder_type.classes_)
		
		pass 

	def featurize_reqs( self , reqs ):

		#output = np.zeros( ( len( reqs ) , self.final_size   ) )
		mapp = {}
		output = []
		i = 0 
		for req in reqs:

			if "text" in req:
				emb = self.featurize( req )
				output.append( emb )
				mapp[req["id"]] = i
				i = i + 1 

		output = np.array( output ) 
		print( "shape embeddings" ,  output.shape )
		return output , mapp 

	def featurize( self , req ):
		# build the features for a single request 
		#print( req["text"])
		emb_text = self.get_average_embedding( req["text"] ) 
		emb_name = self.get_average_embedding( req["name"] ) 
		emb_comment = self.get_comments_embeddings( req )
		emb_components = self.get_components_embeddings( req )

		#emb_mix = (emb_text + emb_name +emb_comment)/3.0

		# those arent word embedding but one-hot encoded vectors
		if "status" in req:
			emb = self.encoder_status.transform(   [req["status"]]   )
			#print(emb)
			emb_status = np.zeros( len(self.encoder_status.classes_)  )
			emb_status[ emb[0] ] = 1.0 

		else:
			emb_status = np.zeros( len(self.encoder_status.classes_)  )

		if "requirement_type" in req:
			emb = self.encoder_type.transform( [req["requirement_type"]]  ) 
			emb_type = np.zeros( len(self.encoder_type.classes_)  )
			emb_type[ emb[0] ] = 1.0 
		else:
			emb_type = np.zeros( len(self.encoder_type.classes_)  )


		#final_embedding = np.hstack( [ emb_text , emb_name , emb_comment , emb_mix ]  )
		final_embedding = 0.1*emb_name + 0.5*emb_text  + 0.3*emb_comment + 0.1*emb_components
		#print( emb_name )
		return final_embedding


	def get_average_embedding(self , txt ):
		# calculates de average embedding
		#embedding = np.zeros( (self.dim) )

		if txt is None or txt is "" or txt is " ":

			return np.zeros( (self.dim) ) # np.zeros( (self.dim))


		txt = txt.lower() 
		words = txt.split(" ")
		embedding = np.zeros( ( self.dim))
		for w in words:
			emb = None 
			try: 
			#if w in model.keys():
			    emb = self.words_model[w]
			except:
			    emb = np.zeros( (self.dim) )
		    
			embedding += emb

		embedding = embedding / len( words )
		return embedding

	def get_comments_embeddings( self , req ):

		if "comments" not in req  or len(  req["comments"]) == 0 :
			#print("noooooooooooooooo")
			return np.zeros( (self.dim) )


		embs = np.zeros( ( self.dim ))
		for comment in req["comments"]:
			txt = comment["text"]
			emb = self.get_average_embedding( txt    )
			embs += emb 

		embs = embs/(len(  req["comments"] + 1 ))

		return embs  


	def get_components_embeddings( self ,  req ,  ):
		if "requirementParts" not in req or -7 not in req["requirementParts"]:
			return np.zeros( (self.dim))
		components_dict = req["requirementParts"][-1]
		if "text" not in components_dict.keys():
			return np.zeros( (self.dim))

		text_list = components_dict["text"] #.replace( '"' , "")[1:-1] #.split(",") #  .split('"' )
		embs = np.zeros( (self.dim))
		for word in text_list:
			if word in model.keys():
	 			emb = self.word_model[word]
				embs += emb 
		embs = embs / (len( text_list ) + 1 )
		return embs 







