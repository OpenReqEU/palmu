import numpy as np

import pandas as pandas

import faiss 
import pickle 
import argparse

#import prepare_data
import os

import json
from flask import Flask
from flask import request
from flask import jsonify 
import requests
from celery import Celery 
from celery.signals import after_task_publish,task_success,task_prerun,task_postrun , task_success

from dataManager import DataManager


FAST_TEXT_MODEL = "./data/wordEmbedding/qtmodel_100.bin"
LGB_PATH = "./data/lgb_results"
JSONS_PATH = "./data"
#dm = DataManager( jsons_path = JSONS_PATH , model_fasttext = FAST_TEXT_MODEL  , lgb_path = LGB_PATH , lgb_name = "Concat")


class Palmu():

	def __init__( self , refresh = False  ):

		self.dm = DataManager( jsons_path = JSONS_PATH , model_fasttext = FAST_TEXT_MODEL  , lgb_path = LGB_PATH , lgb_name = "Concat" , refresh = refresh )


	def create_app(self):

		app = Flask( __name__ )

		@app.route("/getRelated", methods=['GET'])
		def main():

			query_id = request.args.get('id')
			k = request.args.get("k")
			# Multiplies used to enhance the orphan score
			multiplier = 1
			try:
				multiplier = int( request.args.get("m") ) 
			except:
				multiplier = 1 
			if k == None:
				k = 5
			else:
				k = int( k )
			if query_id is None:
				return json.dumps( { "dependencies" : [""] } ) 

		    #Query issues from given id

			similar_issues = self.dm.find_by_id( query_id , k = k , multiplier = multiplier  )
			results = dict()
			results["dependencies"] = similar_issues
			return json.dumps( results )

		@app.route("/newIssue" , methods = ["POST"])
		def new_issue():

			# read request, the requirement
			req = request.get_json()

			#return jsonify( { "dependencies" : [4]*4   })
			k = None 
			try:
				k = int( req["k"] ) 
			except:
				k = 5

			if k == None:
				k = 5
			else:
				k = int( k )
			similar_issues = self.dm.find_by_new( req , k  )

			if similar_issues == []:

				return jsonify( { "dependencies" : [""]})
			else:
		 		return jsonify( {"dependencies": similar_issues })
		    

		@app.route("/postProject", methods=['POST'])
		def post_project():

			#get data from request
		    data = request.get_json()

		    if data is None:
		        return jsonify( {"status" :  "ok"} )


		    project_name = data["projects"][0]["id"]
		    #print("New project: ", project_name)
		    filename = project_name + '.json'
		    self.dm.delete_files()
		    path = os.path.join( JSONS_PATH , filename)

		    with open(path, 'w' , encoding = "utf-8") as json_file:
		        json.dump(data, json_file)
		        json_file.close()

		    # This will re launch the server and reload all available projects
		    data = { "status" : "ok"}

		    return jsonify( data )

		@app.route("/updateRequirements", methods=['POST'])
		def update_reqs():

			data = request.get_json()
			reqs = data["requirements"]
			# Requirements will be added, updated as needed but one important thing is that it will not return 
			# any response until the whole thing is done. so it will only make sense if the number of requeriments to be updated is small
			self.dm.add_or_update_reqs( reqs )
			resp = { "status" : "ok" }
			return jsonify( resp )

		return app 


#celery.control.purge()


	#print( files_json )

	#app.run(host='0.0.0.0' , port=9210 , extra_files = files_json )
	#app.before_first_request( prepare_data.oad_projects()  )
