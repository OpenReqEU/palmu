
import pandas as pd 
import numpy as np
from sklearn import preprocessing

import json
import os
import pickle 
import tables
import faiss 
import featurizer , fastTextUtils , gbmModel 

class DataManager():


	def __init__(self , gloveFile = "./path" , emb_dim = 200 , model_fasttext = "" , lgb_path = "" , lgb_name = "Concat" ):

		# create model. 

		#self.model_glove = self.loadGloveModel( gloveFile )
		self.model_fasttext = fastTextUtils.FastTextUtils( model_fasttext )
		self.emb_dim = self.model_fasttext.dim 
		self.model_lgbm = gbmModel.GBMModel( path = lgb_path , name = lgb_name )

		self.process_files( gloveFile  )
		print("aaasdadas")
		self.indexSize = 0 
		self.buildIndex()
		self.indexSize = self.data.shape[0] - 1 
		
		#l = self.find_by_id( "QTWB-30" )
		#print(l)
		self.test_accuracy()
		return None

	def buildIndex( self ):
		#builds the search index 

		D = self.featurizer.final_size
		#print( self.data.shape  ) 

		#print( self.data.shape   )
		#self.index = faiss.IndexFlatL2( D )
		self.index = faiss.IndexFlatIP( D )
		#self.index.train(  normalize_L2( self.) )
		#cont = np.ascontiguousarray( self.data )
		#cont = faiss.normalize_L2( cont )
		#print(cont)
		self.index.train( self.norm_vec(self.data) )
		self.index.add( self.norm_vec( self.data)  )
		
		print( "indexxxx" , self.index.is_trained) 

	def norm_vec( self , a ):


		a = a / np.sqrt( (a*a).sum(axis = 1 ) ).reshape( a.shape[0]  , 1 )

		return  np.nan_to_num( a )  
	def find_by_id( self , qtid  , k = 5 , k2 = 10  ):
		# return list of know issues 
		if not qtid in self.mappings:
			#print( "ID NOT FOUND")

			return []
		# ind, index od the vector 
		ind = self.mappings[ qtid ] 
    # got the vector

		vector = self.data[ ind , : ].reshape( (1 , self.featurizer.final_size  ))
		#print( vector.shape )
		distances , I = self.index.search( self.norm_vec( vector )   , k )

		#print( I )
		found_issues = []

		data_lgb = np.zeros(   (  len( I[0][1:]  ) , 2*self.emb_dim   ))
		i = 0
		partial_map = {}
		for issue in I[0][1:] :

			#print( self.inverse_mapping[issue] )
			# issue is an index
			emb_candidate = self.data[ issue , : ].reshape( 1 ,self.emb_dim )

			data_point = np.hstack( [ vector , emb_candidate ] )
			data_lgb[ i , : ] = data_point
			partial_map[ i] = issue 
			i += 1 


		top_indexs = self.model_lgbm.get_top_k( data_lgb , k = k2  )

		for index in top_indexs:
			issue = partial_map[index]
			if issue in self.inverse_mapping:
				found_issues.append( self.inverse_mapping[issue]  )


		return found_issues

	def find_by_new( self , openreqJson ):

		# this assumes a openredJson with the fields:
		newId = openreqJson["id"]

		if newId in self.mappings: # no es nuevo en realidad

			return self.find_by_id( newId )

		#embedding = self.get_single_embedding( openreqJson )
		embedding = self.featurizer.featurize( openreqJson )
		if embedding is None :
			# Something happend and
			return []
		else:
			self.add_new_embedding_index( embedding , newId  )
			issues = self.find_by_id( newId )

			return issues

	def add_new_embedding_index( self , embedding , newId ) :

		self.data_elastic.append( embedding )
		self.data = self.hdf5_file.root.data[:]
		self.data = np.array( self.data ).astype( np.float32 )
		self.data = self.data.reshape( ( -1 , self.featurizer.final_size ))

		#rebuild index 
		self.index = faiss.IndexFlatL2( self.emb_dim  )
		self.index.add( self.data )

		self.indexSize = self.indexSize + 1 
		self.mappings[ newId ] = self.indexSize 
		self.inverse_mapping[ self.indexSize ] = newId 

		pickle.dump( self.mappings ,   open( "./data/mappings200.map", "wb" ) , protocol=2 )


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
	    	self.featurizer = pickle.load( open("./data/featurizer.ft" , "rb"))
	    	self.loadHDF5()
	    	print( "File already exists ! loaded ")

	    else:


			files = os.listdir( "./data/")
			files_json = [ "./data/"+f for f in files if ".json" in f ]
			print("aaasdadas")
			embs , mapp = self.get_embeddings( files_json  )
			self.mappings = mapp
			self.inverse_mapping = {v: k for k, v in self.mappings.items() }

			embs = np.array( embs )
			print("wtf men  ")
			print("asdads")
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
		pickle.dump( self.featurizer ,   open( "./data/featurizer.ft", "wb" ) , protocol=2 )

		return all_embeddings , mapping 

	def get_reqs( self , file ):
	    
		data = ""
		with open( file , "r") as f:
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







