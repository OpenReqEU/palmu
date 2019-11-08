
import pandas as pd 
import numpy as np
from sklearn import preprocessing

import json
import os
import pickle 
import tables
import faiss 
import featurizer , fastTextUtils , gbmModel 
import ast
import time 

from tqdm import tqdm 
class DataManager():


	def __init__(self , jsons_path = "./data" , emb_dim = 200 , model_fasttext = "" , lgb_path = "" , lgb_name = "Concat" , refresh = False ):

		# This is the main class of the palmu module. It holds the necesary methods to build the models, make the queries and so on
		self.model_fasttext = fastTextUtils.FastTextUtils( model_fasttext )
		self.emb_dim = self.model_fasttext.dim
		# load GBM models, this will be used for prediction of probabibilities later on, the code assumes the existence of pretrained
		# models existing in the lgb_path folder, the pretrained models must contain in their name, the word pass name parameter
		self.model_lgbm = gbmModel.GBMModel( path = lgb_path , name = lgb_name )
		# creating auxiliar paths 
		self.jsons_path = jsons_path
		self.hdf_path = self.jsons_path + "/hdf_emb.h5"
		self.hdf5_file = None 
		self.mappings_path = self.jsons_path +  "/mappings200.map"
		self.featurizer_path = self.jsons_path + "/featurizer.ft"

		#self.featurizer = pickle.load( open( self.featurizer_path , "rb"))
		self.featurizer = featurizer.Featurizer( self.model_fasttext.model , self.model_fasttext.dim  )
		pickle.dump( self.featurizer ,   open( self.featurizer_path , "wb" ) , protocol=2 )
		
		self.dependencies_dict_path = self.jsons_path + "/dependencies_dict.bin"
		self.dependencies_dict = {}
		start = time.time()
		self.load_projects2( refresh = refresh )
		end = time.time()
		time_difference = end - start
		print("Time diff:" , time_difference )

		return None

	def delete_files(self):

		if os.path.exists( self.hdf_path  ):
			if self.hdf5_file is not None:
				self.hdf5_file.close()
			os.remove( self.hdf_path )
		if os.path.exists( self.mappings_path ):
			os.remove( self.mappings_path )
		if os.path.exists( self.featurizer_path):
			os.remove( self.featurizer_path )

	def load_projects2(self , refresh = False ):

		self.process_files( refresh = refresh )
		if self.ready:
			self.load_HDF5()
			self.indexSize = 0 
			self.build_index()
			self.indexSize = self.data.shape[0] - 1 

	def build_index( self ):
		#builds the search index 

		# Dimenstion of the vectors
		D = self.featurizer.final_size
		self.index = faiss.IndexFlatIP( D )
		self.index.train( self.norm_vec(self.data) )
		self.index.add( self.norm_vec( self.data)  )
		
		print( "Index Trained" , self.index.is_trained) 


	def norm_vec( self , a ):
		# function to normalize vectors 

		a = a / np.sqrt( (a*a).sum(axis = 1 ) ).reshape( a.shape[0]  , 1 )

		return  np.nan_to_num( a )  

	def prune_index( self , I  , qtid ):
		# prune the index to remove the already know dependencies.
		new_index = []
		n1 = len( I[0][1:] )

		for issue in I[0][1:]:
			if issue in self.inverse_mapping.keys():

				proposed_id = self.inverse_mapping[ issue ]
				if qtid not in self.dependencies_dict.keys():
					new_index.append ( issue )
					continue

				if proposed_id in self.dependencies_dict[ qtid ]:
					continue
				else:
					new_index.append( issue )
		n2 = len( new_index )
		diff = n1 - n2 
		print(  "Pruned ids" , diff  )
		return new_index



	def find_by_id( self , qtid  , k = 5 , k2 = 20 , multiplier = 1   ):
		# return list of know issues 

		# if the id is not in the index, return an empty list 
		if not qtid in self.mappings:

			return []
		# ind, index od the vector 
		index_id = self.mappings[ qtid ] 

		vector = self.data[ index_id , : ].reshape( (1 , self.featurizer.final_size  ))
		#print( vector.shape )
		distances , I = self.index.search( self.norm_vec( vector )   , k )

		new_index = self.prune_index( I , qtid )
		#print( I )
		found_issues = []

		# prepare data for the GBM models 
		data_lgb = np.zeros(   (  len( I[0][1:]  ) , 2*self.featurizer.final_size   ))
		i = 0
		partial_map = {}
		for issue in new_index :

			# issue is an index
			emb_candidate = self.data[ issue , : ].reshape( 1 , self.featurizer.final_size )

			data_point = np.hstack( [ vector , emb_candidate ] )
			data_lgb[ i , : ] = data_point
			partial_map[ i] = issue 
			i += 1 


		top_indexs , scores = self.model_lgbm.get_top_k( data_lgb , k = k2  )

		print( partial_map )
		for index , score  in zip( top_indexs , scores ) :

			issue = partial_map[index]
			if issue in self.inverse_mapping:
				json_obj = self.parse_issue( qtid , self.inverse_mapping[issue] , score , multiplier    )

				found_issues.append( json_obj  )

		# return the list of found ids 
		return found_issues	

	def parse_issue( self , qtid , dup , score = "" , multiplier = 1.0  ):


		if dup not in self.dependencies_dict.keys():

			score = score*multiplier


		results = {}
		results["created_at"] = "0"
		results["dependency_type"] = "SIMILAR"
		results["dependency_score"] = str(score) 
		results["description"] = ["palmu"]
		results["fromid"] = qtid
		results["id"] = "{}_{}_SIMILAR".format( qtid , dup )
		results["status"] = "PROPOSED"
		results["toid"] = dup 


		return results

	def add_or_update_reqs(self , list_new_reqs ):

		if self.hdf5_file is not None:
			self.hdf5_file.close() # for safety 
			self.hdf5_file = tables.open_file( self.hdf_path , mode = "r+") # re open

		#
		i = 0 
		print("updating requirements:")
		for req in tqdm( list_new_reqs ):

			query_id = req["id"]
			embedding = self.featurizer.featurize( req )
			embedding = embedding.reshape( ( 1 , 100 ))
			embedding = self.norm_vec( embedding )

			if query_id in self.mappings.keys():
				# Idd already exists in dataset
				#
				indexId = self.mappings[query_id]

				# after we get the embedding
				self.hdf5_file.root.data[ indexId , : ] = embedding

			else:
				#
				self.hdf5_file.root.data.append( embedding )
				newIndex = len( self.hdf5_file.root.data[:] ) - 1 
				#print( newIndex )
				self.mappings[query_id] = newIndex
			i += 1  

		# save the modified mappings
		files = os.listdir( self.jsons_path )
		files_json = [ self.jsons_path+"/"+f for f in files if ".json" in f ]
		if len( files_json) == 0:
			self.ready = False 
			return False 
		self.dependencies_dict = self.get_dependencies_dict(  files_json )

		pickle.dump( self.dependencies_dict , open( self.dependencies_dict_path ,"wb") , protocol = 2 )
		pickle.dump( self.mappings ,   open( self.mappings_path, "wb" ) , protocol=2 )
		print("updates")
		print( "number of keys:" , len( self.mappings.keys() ))
		print( "number of reqss:" , len( self.hdf5_file.root.data[:]))
		self.hdf5_file.close()
		self.load_HDF5()
	

		return True
	
	def find_by_new( self , openreqJson , k = 1000 , k2 = 11  ):

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
			
			self.add_or_update_reqs( [ openreqJson] )

			issues = self.find_by_id( newId  , k , k2 )

			return issues

	def process_files( self , refresh = False   ):
		# this function saves on disk the mappings in between the vector embeddings and the 
		# List existing files on data folder , 

		if refresh: 
			self.delete_files()

		if os.path.isfile( self.hdf_path ):

			self.mappings = pickle.load( open( self.mappings_path , "rb") )
			self.inverse_mapping = {v: k for k, v in self.mappings.items()}
			self.featurizer = pickle.load( open( self.featurizer_path , "rb"))
			self.dependencies_dict = pickle.load( open( self.dependencies_dict_path , "rb"))
			#print( self.dependencies_dict )
			self.load_HDF5()
			print( "File already exists ! loaded ")
			self.ready = True 


		else:


			files = os.listdir( self.jsons_path )
			files_json = [ self.jsons_path+"/"+f for f in files if ".json" in f ]

			if ( len(files_json) == 0 ):
				# there are no files to build,  do nothing. 
				self.ready = False 
				return 


			#print("Processing Json Files")
			embs , mapp = self.get_embeddings( files_json  )
			self.dependencies_dict = self.get_dependencies_dict( files_json )
			self.mappings = mapp
			self.inverse_mapping = {v: k for k, v in self.mappings.items() }

			# load an embeddings array 

			embs = np.array( embs )
			# save mappings to disk 
			pickle.dump( mapp ,   open( self.mappings_path, "wb" ) , protocol=2 )
			pickle.dump( self.dependencies_dict , open( self.dependencies_dict_path ,"wb") , protocol = 2 )


			hdf5_embedd_file = tables.open_file(  self.hdf_path , mode='w')
			a = tables.Atom.from_dtype( np.dtype('<f8'), dflt=0.0 )
			shape = ( 0 ,100 )
			earray = hdf5_embedd_file.create_earray( hdf5_embedd_file.root ,'data', a ,shape,"Embeddings")
			#print("*"*3)
			#print( earray.nrows )
			#print( earray.rowsize)
			#print( earray.atom )
			for emb in embs:
				#print("adasdasda")
				#print( emb.shape )
				emb = emb.reshape( (1 , -1) )
				earray.append( emb )


			hdf5_embedd_file.close()
			self.ready = True 
			

			print("HDF5 FILE CREATED AND LOADED")
			return 
			
	def get_dependencies_dict( self , files_json  ):

		total_deps = []

		for file in files_json:

			deps = self.get_deps( file )
			total_deps = total_deps + deps 

		deps_dict = {}
		for d in total_deps:
			fromid = d["fromid"]
			deps_dict[fromid] = []


		for d in total_deps:
			fromid = d["fromid"]
			toid = d["toid"]
			deps_dict[fromid].append( toid )


		return deps_dict 


	def load_HDF5( self ):

		self.mappings = pickle.load( open( self.mappings_path , "rb") )
		self.inverse_mapping = {v: k for k, v in self.mappings.items()}
		self.featurizer = pickle.load( open( self.featurizer_path , "rb"))
		
		f = tables.open_file( self.hdf_path , mode = "a")
		self.hdf5_file = f
		self.data_elastic = self.hdf5_file.root.data
		print("#"*10)
		print( self.data_elastic.rowsize)
		print( self.data_elastic.nrows )

		self.data = self.hdf5_file.root.data[:]
		self.data = np.array( self.data ).astype( np.float32 )
		self.data = self.data.reshape( ( -1 , self.featurizer.final_size ))

		

	def get_embeddings( self ,  files_json  ):
		# return the 
		# id - > embeddings correspondence
		
		all_embeddings = []
		index = 0
		mapping = {}
		#print( files_json )

		all_reqs = []
		for file in files_json:
			#print( file )
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
		#encoder_status = preprocessing.LabelEncoder()
		#encoder_type = preprocessing.LabelEncoder()
		#encoder_status = encoder_status.fit( status )
		#encoder_type = encoder_type.fit( types )
		self.featurizer = featurizer.Featurizer( self.model_fasttext.model , self.model_fasttext.dim  )

		all_embeddings , mapping = self.featurizer.featurize_reqs( all_reqs )
		pickle.dump( self.featurizer ,   open( self.featurizer_path , "wb" ) , protocol=2 )

		return all_embeddings , mapping 


	def get_reqs( self , file ):
	    
		data = ""
		with open( file , "r" , encoding = "utf-8") as f:
			data = f.read()

			data = json.loads( data )
			#print(" getting requirements from {}  - number of reqs: {}".format(  file , len(data["requirements"])) )
		return data["requirements"]

	def get_deps( self , file ):

		data = ""
		with open( file , "r" , encoding = "utf-8") as f:
			data = f.read()

			data = json.loads( data )
			#print(" getting requirements from {}  - number of reqs: {}".format(  file , len(data["requirements"])) )
		return data["dependencies"]




