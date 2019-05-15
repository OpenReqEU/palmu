
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
		self.process_files( gloveFile  )
		print("aaasdadas")
		self.indexSize = 0 
		self.buildIndex()
		self.indexSize = self.data.shape[0] - 1 

		#l = self.find_by_id( "QTWB-30" )
		f = open("./test.txt" , "r")
		d = f.read()
		f.close()

		req = json.loads( d )
		l = self.find_by_new( req )
		print( l )
		return None

	def buildIndex( self ):
		#builds the search index 

		D = self.emb_dim
		print( self.data.shape  ) 
		self.index = faiss.IndexFlatL2( D )
		self.index.add( self.data )

	def find_by_id( self , qtid  , k = 5 ):
		# return list of know issues 
		if not qtid in self.mappings:
			print( "ID NOT FOUND")

			return None

		ind = self.mappings[ qtid ]
    # got the vector

		vector = self.data[ ind , : ].reshape( (1 , self.emb_dim ))
		print( vector.shape )
		distances , I = self.index.search( vector , k )

		print( I )
		found_issues = []

		for issue in I[0][1:] :

			found_issues.append( self.inverse_mapping[issue]  )
		return found_issues

	def find_by_new( self , openreqJson ):

		# this assumes a openredJson with the fields:
		newId = openreqJson["id"]

		if newId in self.mappings: # no es nuevo en realidad

			return self.find_by_id( newId )

		embedding = self.get_single_embedding( openreqJson )
		if embedding is None :
			# Something happend and
			return "Invalid req"
		else:
			self.add_new_embedding_index( embedding , newId  )
			issues = self.find_by_id( newId )

			return issues

	def add_new_embedding_index( self , embedding , newId ) :



		self.data_elastic.append( embedding )
		self.data = self.hdf5_file.root.data[:]
		self.data = np.array( self.data ).astype( np.float32 )
		self.data = self.data.reshape( ( -1 , self.emb_dim ))

		#rebuild index 
		self.index = faiss.IndexFlatL2( self.emb_dim  )
		self.index.add( self.data )

		self.indexSize = self.indexSize + 1 
		self.mappings[ newId ] = self.indexSize 
		self.inverse_mapping[ self.indexSize ] = newId 

		pickle.dump( self.mappings ,   open( "./data/mappings200.map", "wb" ) , protocol=2 )


	def get_single_embedding( self , req ):

		if "text" in req.keys():
#print( req["text"])
			name_emb = self.get_embedding_txt( req["name"] , self.model_glove )
			embedding = self.get_embedding_txt( req["text"]  , self.model_glove )
			comment_emb = self.get_embedding_com( req , self.model_glove )
			component_emb = self.get_embedding_components( req , self.model_glove )
			embedding =  0.1*name_emb + 0.5*embedding + 0.3*comment_emb + 0.1*component_emb 

			return embedding 
		else: 
			return None


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

	def process_files( self , gloveFile ):
	    # this function saves on disk the mappings in between the vector embeddings and the 
	    # List existing files on data folder , 

	    if os.path.isfile( "./data/hdf_emb.h5" ):

	    	self.mappings = pickle.load( open( "./data/mappings200.map" , "rb") )
	    	self.inverse_mapping = {v: k for k, v in self.mappings.items()}
	    	self.loadHDF5()
	    	print( "File already exists ! loaded ")

	    else:


			files = os.listdir( "./data/")
			files_json = [ "./data/"+f for f in files if ".json" in f ]
			print("aaasdadas")
			embs , mapp = self.get_embeddings( files_json  )
			self.mappings = mapp
			self.inverse_mapping = {v: k for k, v in self.mappings.items()}

			embs = np.array( embs )
			print("wtf men ")

			pickle.dump( mapp ,   open( "./data/mappings200.map", "wb" ) , protocol=2 )
			#np.save( "./data/embbedings200.npy" , embs   )

			hdf5_embedd_file = tables.open_file(  "./data/hdf_emb.h5" , mode='w')
			a = tables.Atom.from_dtype( np.dtype('<f8'), dflt=0.0 )
			shape = ( 0 , )
			earray = hdf5_embedd_file.create_earray(hdf5_embedd_file.root,'data', a ,shape,"Embeddings")

			for emb in embs:

				earray.append( emb )


			hdf5_embedd_file.close()
			self.loadHDF5()

			print("HDF5 FILE CREATED AND LOADED")
			#atom=tb.StringAtom(itemsize=8)

	def loadHDF5( self ):

		f = tables.open_file( "./data/hdf_emb.h5" , mode = "a")
		self.hdf5_file = f
		self.data_elastic = self.hdf5_file.root.data 
		self.data = self.hdf5_file.root.data[:]
		self.data = np.array( self.data ).astype( np.float32 )
		self.data = self.data.reshape( ( -1 , self.emb_dim ))



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
					if req["id"] == "QTBUG-6229":

						data = json.dumps( req )
						f = open( "test.txt" , "w") 
						f.write(data)
						f.close()
						print( "Saved test requ")
						continue
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
