
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


class DataManager():


	def __init__(self , jsons_path = "./data" , emb_dim = 200 , model_fasttext = "" , lgb_path = "" , lgb_name = "Concat" ):

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
		self.dependencies_dict_path = self.jsons_path + "/dependencies_dict.bin"
		self.dependencies_dict = {}
		self.load_projects2( refresh = False )
		#self.process_files()
		#self.test_accuracy()
		self.test_update()
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
		self.loadHDF5()
		self.indexSize = 0 
		self.build_index()
		self.indexSize = self.data.shape[0] - 1 

	def build_index( self ):
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


		top_indexs , scores = self.model_lgbm.get_top_k( data_lgb , k = k2  )


		for index , score  in zip( top_indexs , scores ) :
			issue = partial_map[index]
			if issue in self.inverse_mapping:
				json_obj = self.parse_issue( qtid , self.inverse_mapping[issue] , score   )

				found_issues.append( json_obj  )

		# return the list of found ids 
		return found_issues	

	def parse_issue( self , qtid , dup , score = "" , multiplier = 1  ):


		if dup in self.dependencies_dict.keys():

			score = score
		else:
			print( "ORPHAN FOUND" , dup )
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

		self.hdf5_file.close() # for safety 
		self.hdf5_file = tables.open_file( self.hdf_path , mode = "r+") # re open

		#data = self.hdf5_file.root.data[:]
		i = 0 

		for req in list_new_reqs:

			idd = req["id"]
			embedding = self.featurizer.featurize( req )
			embedding = embedding.reshape( ( 1 , 100 ))
			embedding = self.norm_vec( embedding )

			if idd in self.mappings.keys():
				# Idd already exists in dataset
				print(idd , i  , " in index ")
				indexId = self.mappings[idd]

				# after we get the embedding
				self.hdf5_file.root.data[ indexId , : ] = embedding

			else:
				print(idd , i , "not in index " )
				self.hdf5_file.root.data.append( embedding )
				newIndex = len( self.hdf5_file.root.data[:] ) - 1 
				print( newIndex )
				self.mappings[idd] = newIndex
			i += 1  

		# save the modified mappings 
		pickle.dump( self.mappings ,   open( self.mappings_path, "wb" ) , protocol=2 )
		print("updates")
		print( "number of keys:" , len( self.mappings.keys() ))
		print( "number of reqss:" , len( self.hdf5_file.root.data[:]))
		self.hdf5_file.close()
		self.loadHDF5()
	

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
			
			embedding = embedding.reshape( ( 1 , 100 ))
			print( embedding.shape )
			embedding = self.norm_vec( embedding )
			self.add_new_embedding_index( embedding , newId  )
			issues = self.find_by_id( newId  , k , k2 )

			return issues

	def add_new_embedding_index( self , embedding , newId ) :

		#prit( self.data_elastic  )
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
		pickle.dump( self.mappings ,   open( self.mappings_path , "wb" ) , protocol=2 )


	def process_files( self , refresh = False   ):
		# this function saves on disk the mappings in between the vector embeddings and the 
		# List existing files on data folder , 

		#if refresh:
			# 
		#	self.delete_files()

		if os.path.isfile( self.hdf_path ):

			self.mappings = pickle.load( open( self.mappings_path , "rb") )
			self.inverse_mapping = {v: k for k, v in self.mappings.items()}
			self.featurizer = pickle.load( open( self.featurizer_path , "rb"))
			self.dependencies_dict = pickle.load( open( self.dependencies_dict_path , "rb"))
			#print( self.dependencies_dict )
			self.loadHDF5()
			print( "File already exists ! loaded ")

		else:


			files = os.listdir( self.jsons_path )
			files_json = [ self.jsons_path+"/"+f for f in files if ".json" in f ]
			print("Processing Json Files")
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
			#self.loadHDF5()

			print("HDF5 FILE CREATED AND LOADED")
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


	def loadHDF5( self ):

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

		print("loadddeddd stufff")

	def get_embeddings( self ,  files_json  ):
		# return the 
		# id - > embeddings correspondence
		
		all_embeddings = []
		index = 0
		mapping = {}
		print( files_json )

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
			#print(" getting requirements from {}  - number of reqs: {}".format(  file , len(data["requirements"])) )
		return data["requirements"]

	def get_deps( self , file ):

		data = ""
		with open( file , "r" , encoding = "utf-8") as f:
			data = f.read()

			data = json.loads( data )
			#print(" getting requirements from {}  - number of reqs: {}".format(  file , len(data["requirements"])) )
		return data["dependencies"]
	        
	def test_update( self ): 

		#
		reqs = self.get_reqs( "./data/QTWB.json") 
		
		self.add_or_update_reqs( reqs )
		for req in reqs:
			req["text"] = req["text"] + " modified dfdsfsdfsfs"

		# test modify content same req 
		self.add_or_update_reqs( reqs )

		for  i , req in enumerate( reqs ) :
			req["id"]  = req["id"] + "___{}".format( i*121 )

		#print( reqs[0]["id"])
		self.add_or_update_reqs(reqs )


	def test_accuracy( self ):

		df = pd.read_csv("./dataset_palmu_test_duplicates.csv")
		df = df[:]
		ids = df["ids"].values
		dependencies = df["dependencies"].values
		print("TEST")

		results = []

		l = 0

		ks = [ 100 , 1000  ]

		k100_tp = []

		true_positives_dict = {}
		true_positives_dict[ ks[0] ] = 0.0
		true_positives_dict[ ks[1] ] = 0.0

		total_deps =  0
		for idd , dep in zip(ids , dependencies):
			#print( type( dep ))

			total_deps += len( ast.literal_eval( dep )  )

		for idd , dep  in zip(ids , dependencies )  :
			#if l % 500 == 0 :

			print( l )
			l = l + 1

			corrects_by_k = []

			avg_true_positives = 0.0
			

			for k  in ks :

				issues = self.find_by_id( idd  , k = k  , k2 = k )

				corrects = 0
				correct_issues = []
				for i in issues: 

					#print( type(i) )
					id_issue = i["toid"]

					#print( type( dep )  )
					if id_issue in dep:
						corrects = corrects + 1
						correct_issues.append( i["toid"] )

				d = 0
				#print( len(dep) )
				#print( corrects )
				#true_positives_rate = corrects/float( len( dep  )  + d )
				true_positives_dict[ k ] += corrects  

				#print(" True positives rate: " ,  true_positives_rate )
				#print( "False positives rate:" , 1 - true_positives_rate )
				#print( "Samples:" , len(issues))
				corrects_by_k.append( correct_issues )

			results.append( corrects_by_k )


		df["palmu_depencendies"] = results 

		df2 = pd.DataFrame( df.palmu_depencendies.to_list() , columns = [ "k{}".format(x) for x in ks ] )

		df_final = pd.concat( [ df[ ["ids" , "dependencies" ] ] , df2 ] , axis = 1 )

		df_final.to_csv("./results_test_k100_k1000_with_lgb_complete.csv")

		k5_found = 0
		k20_found = 0

		for k in ks: 
			print( total_deps )
			print( true_positives_dict[k] )
			arr =  (true_positives_dict[k]/total_deps)*100

			print("Average true positives rate for {} candidates: {} ".format( k , arr)  )





