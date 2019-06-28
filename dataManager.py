
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

"""
[
  {
    "created_at": 0,
    "dependency_score": 0,
    "dependency_type": "CONTRIBUTES",
    "description": [
      "string"
    ],
    "fromid": "string",
    "id": "string",
    "status": "PROPOSED",
    "toid": "string"
  }
"""

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
		self.hdf5_file = None 
		self.mappings_path = self.jsons_path +  "/mappings200.map"
		self.featurizer_path = self.jsons_path + "/featurizer.ft"
		#self.milla_url = "http://0.0.0.0:9203" #/otherDetectionService"
		#self.palmu_url = "http://0.0.0.0:9210" # /postProjec"
		#self.mallikas_url = "https://api.openreq.eu/mallikas"

		#self.post_to_milla( code ="ok" )
		#print("NICe")
		self.delete_files()
		self.load_projects2( refresh = True )
		#self.process_files()
		#self.test_accuracy()
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


	def post_to_milla( self , code = "ok" ):

		params = { "status":  code  }
		r = requests.post( url = self.milla_url + "/palmuInterface" , data = params    )

		return r 


	def get_projects(self ):

		return  [ "QTWB"]  #projects[:1]

	def load_projects2(self , refresh = False ):

		self.process_files( refresh = refresh )
		self.loadHDF5()
		self.indexSize = 0 
		self.buildIndex()
		self.indexSize = self.data.shape[0] - 1 

	def load_projects( self ):
		#self.load_from_milla( "QTWB" )
		#return True 
		# get project 
		for project in self.get_projects():
			print( "Loading project: {}".format( project ) )

			self.load_from_milla( project )

		self.process_files( refresh = True )

		print( "index building")

		self.indexSize = 0 
		self.buildIndex()
		self.indexSize = self.data.shape[0] - 1 

		print (" index adsasd")
	def load_from_milla( self , projectId   ):
		# url : url to palmu 
		# projectId 
		#headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:61.0) Gecko/20100101 Firefox/61.0'} 
		params = { "projectId":  projectId , "url" : self.palmu_url + "/postProject" }
		#headers = {'content-type': 'application/json' }
		print("requeeeest")
		print( params )
		r = requests.post( url = self.milla_url + "/otherDetectionService" , data = params    )
		print( r.json() )
		
		# milla will send the data to the given url 
		
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


		top_indexs , scores = self.model_lgbm.get_top_k( data_lgb , k = k2  )


		for index , score  in zip( top_indexs , scores ) :
			issue = partial_map[index]
			if issue in self.inverse_mapping:
				json_obj = self.parse_issue( qtid , self.inverse_mapping[issue] , score   )

				found_issues.append( json_obj  )

		# return the list of found ids 
		return found_issues

	def old_implementation( self  , indxs , distances , qtid   ):


		for  i , issue in  enumerate( indxs[0][:1] ):
			json_obj = self.parse_issue( qtid , self.inverse_mapping )

			


	def parse_issue( self , qtid , dup , score = ""  ):

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

		if refresh:
			# 
			self.delete_files()

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
			shape = ( 0 ,100 )
			earray = hdf5_embedd_file.create_earray(hdf5_embedd_file.root,'data', a ,shape,"Embeddings")
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

	def add_new_project_from_file( self , filename):


		return True

	def get_reqs( self , file ):
	    
		data = ""
		with open( file , "r" , encoding = "utf-8") as f:
			data = f.read()

			data = json.loads( data )
			#print(" getting requirements from {}  - number of reqs: {}".format(  file , len(data["requirements"])) )
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

				issues = self.find_by_id( idd  , k = k  , k2 = 20 )

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

		"""
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
		""" 


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



