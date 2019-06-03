
import pandas as pd 
import numpy as np
from sklearn import preprocessing

import json
import os
import pickle 
import tables
import faiss 
import featurizer , fastTextUtils , gbmModel 
import requests


class DataManager():


	def __init__(self , jsons_path = "./data" , emb_dim = 200 , model_fasttext = "" , lgb_path = "" , lgb_name = "Concat" ):

		# create model. 

		# load fast text model
		self.model_fasttext = fastTextUtils.FastTextUtils( model_fasttext )
		self.emb_dim = self.model_fasttext.dim
		# load GBM models
		self.model_lgbm = gbmModel.GBMModel( path = lgb_path , name = lgb_name )
		# creating auxiliar paths 
		self.jsons_path = jsons_path
		self.hdf_path = self.jsons_path + "/hdf_emb.h5"
		self.mappings_path = self.jsons_path +  "/mappings200.map"
		self.featurizer_path = self.jsons_path + "/featurizer.ft"


		self.process_files(  )
		self.indexSize = 0 
		self.buildIndex()
		self.indexSize = self.data.shape[0] - 1 
		
		#l = self.find_by_id( "QTWB-30" )
		#print(l)
		#self.test_accuracy()
		self.milla_url = "https://api.openreq.eu/milla/otherDetectionService"
		return None

	def load_from_milla( self , projectId , url ):
		# url : url to palmu 
		# projectId 

		params = { "projectId":  projectId , "url" : url }
		headers = {'content-type': 'application/json'}

		r = requests.get( url = self.milla_url , params = params )

		return True 
	def buildIndex( self ):
		#builds the search index 

		# Dimenstion of the vectors
		D = self.featurizer.final_size
		self.index = faiss.IndexFlatIP( D )
		#self.index.train(  normalize_L2( self.) )
		#cont = np.ascontiguousarray( self.data )
		#cont = faiss.normalize_L2( cont )
		self.index.train( self.norm_vec(self.data) )
		self.index.add( self.norm_vec( self.data)  )
		
		print( "Index Trained" , self.index.is_trained) 

	def norm_vec( self , a ):
		# function to normalize vectors 

		a = a / np.sqrt( (a*a).sum(axis = 1 ) ).reshape( a.shape[0]  , 1 )

		return  np.nan_to_num( a )  

	def find_by_id( self , qtid  , k = 5 , k2 = 20  ):
		# return list of know issues 

		# if the id is not in the index, return an empty list 
		if not qtid in self.mappings:

			return []
		# ind, index od the vector 
		index_id = self.mappings[ qtid ] 

		vector = self.data[ index_id , : ].reshape( (1 , self.featurizer.final_size  ))
		#print( vector.shape )
		distances , I = self.index.search( self.norm_vec( vector )   , k )

		#print( I )
		found_issues = []

		# prepare data for the GBM models 
		data_lgb = np.zeros(   (  len( I[0][1:]  ) , 2*self.featurizer.final_size   ))
		i = 0
		partial_map = {}
		for issue in I[0][1:] :

			#print( self.inverse_mapping[issue] )
			# issue is an index
			emb_candidate = self.data[ issue , : ].reshape( 1 , self.featurizer.final_size )

			data_point = np.hstack( [ vector , emb_candidate ] )
			data_lgb[ i , : ] = data_point
			partial_map[ i] = issue 
			i += 1 


		top_indexs = self.model_lgbm.get_top_k( data_lgb , k = k2  )

		for index in top_indexs:
			issue = partial_map[index]
			if issue in self.inverse_mapping:
				found_issues.append( self.inverse_mapping[issue]  )

		# return the list of found ids 
		return found_issues

	def find_by_new( self , openreqJson , k = 1000 , k2 = 15  ):

		# openredJson must be a valid openreqJson 
		newId = openreqJson["id"]

		# if the ID exists in the mappings run the 
		if newId in self.mappings: 

			return self.find_by_id( newId , k , k2  )

		# Get the embedding for the new json
		embedding = self.featurizer.featurize( openreqJson )

		if embedding is None :
			# is the embedding is null 
			return []
		else:
			embedding = self.norm_vec( embedding )
			self.add_new_embedding_index( embedding , newId  )
			issues = self.find_by_id( newId  , k , k2 )

			return issues

	def add_new_embedding_index( self , embedding , newId ) :

		self.data_elastic.append( embedding )
		self.data = self.hdf5_file.root.data[:]
		self.data = np.array( self.data ).astype( np.float32 )
		self.data = self.data.reshape( ( -1 , self.featurizer.final_size ))

		#rebuild index 
		self.index = faiss.IndexFlatL2( self.featurizer.final_size  )
		self.index.add( self.data )

		self.indexSize = self.indexSize + 1 
		self.mappings[ newId ] = self.indexSize 
		self.inverse_mapping[ self.indexSize ] = newId 

		# update mappings 
		pickle.dump( self.mappings ,   open( self.mappings_path ) , protocol=2 )


	def process_files( self   ):
		# this function saves on disk the mappings in between the vector embeddings and the 
		# List existing files on data folder , 
		if os.path.isfile( self.hdf_path ):

			self.mappings = pickle.load( open( self.mappings_path , "rb") )
			self.inverse_mapping = {v: k for k, v in self.mappings.items()}
			self.featurizer = pickle.load( open( self.featurizer_path , "rb"))
			self.loadHDF5()
			print( "File already exists ! loaded ")

		else:


			files = os.listdir( self.jsons_path )
			files_json = [ self.jsons_path+"/"+f for f in files if ".json" in f ]
			print("Processing Json Files")
			embs , mapp = self.get_embeddings( files_json  )
			self.mappings = mapp
			self.inverse_mapping = {v: k for k, v in self.mappings.items() }

			# load an embeddings array 

			embs = np.array( embs )
			# save mappings to disk 

			pickle.dump( mapp ,   open( self.mappings_path, "wb" ) , protocol=2 )


			hdf5_embedd_file = tables.open_file(  self.hdf_path , mode='w')
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

		f = tables.open_file( self.hdf_path , mode = "a")
		self.hdf5_file = f
		self.data_elastic = self.hdf5_file.root.data 
		self.data = self.hdf5_file.root.data[:]
		self.data = np.array( self.data ).astype( np.float32 )
		self.data = self.data.reshape( ( -1 , self.featurizer.final_size ))



	def get_embeddings( self ,  files_json  ):
		# return the 
		# id - > embeddings correspondence
		
		all_embeddings = []
		comment_embeddings = [] 
		index = 0
		mapping = {}
		print( files_json )

		all_reqs = []
		for file in files_json:
			print( file )
			requirements = self.get_reqs( file )
			all_reqs = all_reqs + requirements 

		status = ["default"]
		types = ["default"]
		for req in all_reqs:
			if "status" in req :
				status.append( req["status"] )

			if "requirement_type" in req:
				types.append( req["requirement_type"])


		# list unique status 
		status = list( set( status))
		types = list(set(types))
		print( status )
		print( types )
		encoder_status = preprocessing.LabelEncoder()
		encoder_type = preprocessing.LabelEncoder()
		encoder_status = encoder_status.fit( status )
		encoder_type = encoder_type.fit( types )
		self.featurizer = featurizer.Featurizer( self.model_fasttext.model , self.model_fasttext.dim  , encoder_status , encoder_type)

		all_embeddings , mapping = self.featurizer.featurize_reqs( all_reqs )
		pickle.dump( self.featurizer ,   open( self.featurizer_path , "wb" ) , protocol=2 )

		return all_embeddings , mapping 

	def get_reqs( self , file ):
	    
		data = ""
		with open( file , "r" , encoding = "utf-8") as f:
			data = f.read()

			data = json.loads( data )
			print(" getting requirements from {}  - number of reqs: {}".format(  file , len(data["requirements"])) )
		return data["requirements"]
	        
	def test( self ) :

		# create some test 

		f = open( "./test_large.txt" , "r") 


		reqs = []
		dupls = []
		print("<sdasdasdadsassssssssssssssssssssss")
		for line in f:

			# each line is a req 
			print(line)
			req = json.loads( line )

			issues = self.find_by_new( req )

			reqs.append( req["id"] )
			dupls.append( issues )

		f.close()
		d = {

			"Ids" : reqs , 
			"Similar issues" : dupls   
		} 


		df = pd.DataFrame( d ) 
		df.to_csv("./results_test.csv")


	def test_accuracy( self ):

		df = pd.read_csv("./dataset_palmu_test_duplicates.csv")
		df = df[:]
		ids = df["ids"].values
		dependencies = df["dependencies"].values


		results = []

		l = 0

		ks = [ 100 , 1000  ]

		k100_tp = []
		for idd , dep  in zip(ids , dependencies )  :
			#if l % 500 == 0 :

			print( l )
			l = l + 1

			corrects_by_k = []
			for k  in ks :

				issues = self.find_by_id( idd  , k = k  )

				corrects = 0
				correct_issues = []
				for i in issues: 

					if i in dep:
						corrects = corrects + 1
						correct_issues.append( i )

				d = 0 
				if len( dep) == 0 :
					d = 1 
				true_positives_rate = corrects/float( len( dep  )  + d )
				
				#print(" True positives rate: " ,  true_positives_rate )
				#print( "False positives rate:" , 1 - true_positives_rate )
				#print( "Samples:" , len(issues))
				corrects_by_k.append( correct_issues )

			results.append( corrects_by_k )


		df["palmu_depencendies"] = results 

		df2 = pd.DataFrame( df.palmu_depencendies.to_list() , columns = [ "k{}".format(x) for x in ks ] )

		df_final = pd.concat( [ df[ ["ids" , "dependencies" ] ] , df2 ] , axis = 1 )

		df_final.to_csv("./results_test_k5_k20_with_lgb_duplicates.csv")

		k5_found = 0
		k20_found = 0
		k100_found = 0 
		k200_found = 0 
		total = 0
		for i , j , k     in zip(df_final["k100"].values , df_final["k1000"] , df_final["dependencies"]  ):
			k5_found += len(i )
			k20_found += len( j )
			total += len( k )
			#k100_found += len(m) 
			#k200_found += len(l )

		total = df_final.shape[0]
		print("OVERAll accuracy")
		print( "K 100 accuracy: {}".format( k5_found/float( total) )  )
		print( "K 1000 accuracy: {}".format( k20_found/float( total) )  )
		#print( "K 100 accuracy: {}".format( k100_found/float( total) )  )
		#print( "K 1000 accuracy: {}".format( k200_found/float( total) )  )

#####################
#### DEPRECATED #####
#####################
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



