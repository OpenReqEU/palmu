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

local_host = "127.0.0.1"



def make_celery( app ):
	celery = Celery(
		app.import_name , 
		backend = app.config["CELERY_RESULT_BACKEND"] ,  
		broker = app.config["CELERY_BROKER_URL"]
		)

	celery.conf.update( app.config )

	class ContextTask( celery.Task ):

		def __call__(self , *args , **kwargs):
			with app.app_context():
				return self.run( *args , **kwargs )

	celery.Task = ContextTask
	return celery


app = Flask(__name__)
app.config.update(
	CELERY_BROKER_URL = "redis://{}:6379".format(local_host  ),
	CELERY_RESULT_BACKEND = "redis://{}:6379".format( local_host ) 

	)

celery = make_celery(app)
#celery.control.purge()




FAST_TEXT_MODEL = "./data/wordEmbedding/qtmodel_100.bin"
LGB_PATH = "./data/lgb_results"

dm = DataManager( jsons_path = "./data" , model_fasttext = FAST_TEXT_MODEL  , lgb_path = LGB_PATH , lgb_name = "Concat")

#### END POINTS AND CELERY TASKS 

@celery.task(  callback = dm.loadHDF5 )
def process_json_files():

	dm.process_files(refresh = True)
	headers = {'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:61.0) Gecko/20100101 Firefox/61.0'} 
	params = { "status":  "200"  }
	headers = {'content-type': 'application/json' }
	print( "cosas chidas")
	r = requests.post( url = "http://{}:9210/load".format( local_host ) , data = params   )
	#dm.post_to_milla("OK")
	print(r)
	return "ok"

@app.route("/testCelery" , methods= ["GET"])
def test():
	result = run_cel.delay( )
	return "Nice"


@app.route("/getRelated", methods=['GET'])
def main():

	idd = request.args.get('id')
	k = request.args.get("k")

	if k == None:
		k = 5
	else:
		k = int( k )
	if idd is None:
		return {}

    #print( "Query issue: " , idd )
	similar_issues = dm.find_by_id( idd , k = k )


	#if similar_issues  == []:
	#	results["similar_issues"] = []
	#	return json.dumps( results )

	#results["similar_issues"] = similar_issues 
	results = dict()
	results["dependencies"] = similar_issues
	return json.dumps( results )

@app.route("/newIssue" , methods = ["POST"])
def new_issue():

	# read request 
	req = request.get_json()
	k = None 
	try:
		k = req["k"]
	except:
		k = 5

	if k == None:
		k = 5
	else:
		k = int( k )
	similar_issues = dm.find_by_new( req , k  )

	if similar_issues == []:

		return {}
      
    
	return json.dumps( similar_issues ) 

@app.route("/postProject", methods=['POST'])
def post_project():

    data = request.get_json()
    #print(data)

    if data is None:
        return jsonify( {"status" :  "ok"} )

    dm.post_to_milla( "working" )

    project_name = data["projects"][0]["id"]

    print("New project: ", project_name)

    filename = project_name + '.json'

    path = os.path.join('./data/', filename)

    with open(path, 'w') as json_file:
        json.dump(data, json_file)
        json_file.close()


    #dm.process_files( path , refresh = True )
    #return "ok"
    data= { "status" : "ok"}
    return jsonify( data )

#app.before_first_request( prepare_data.onstart() )

if __name__ == '__main__':

	#prepare_data.onstart()

	
	print("Palmu main")
	#print( files_json )

	#app.run(host='0.0.0.0' , port=9210 , extra_files = files_json )
	#app.before_first_request( prepare_data.oad_projects()  )


"""
CONTAINER_NAME="$JOB_BASE_NAME"
sleep 10
docker run --rm -d -p 9210:9210 --label "BUILD_NUMBER=$BUILD_NUMBER" --name "$CONTAINER_NAME" -i -t "$CONTAINER_NAME"
"""
