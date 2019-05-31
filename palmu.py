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

from dataManager import DataManager
app = Flask(__name__)

parser = argparse.ArgumentParser(description=' Get similar reports from jira data')

parser.add_argument('--id', type =str , required = True )


FAST_TEXT_MODEL = "./data/wordEmbedding/qtmodel_100.bin"
LGB_PATH = "./data/lgb_results"

dm = DataManager( jsons_path = "./data" , model_fasttext = FAST_TEXT_MODEL  , lgb_path = LGB_PATH , lgb_name = "Concat")

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
	if similar_issues  == []:
		return {}

	return json.dumps(similar_issues)

@app.route("/newIssue" , methods = ["POST"])
def new_issue():

	# read request 
	req = request.get_json()

	k = req["k"]
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
    if data is None:
        return {}


    project_name = data["projects"][0]["id"]

    print("New project: ", project_name)

    filename = project_name + '.json'

    path = os.path.join('./data/', filename)

    print(path)

    with open(path, 'w') as json_file:
        json.dump(data, json_file)
        json_file.close()

    prepare_data.process_files()

    return { "status" : "ok"}


#app.before_first_request( prepare_data.onstart() )

if __name__ == '__main__':

	#prepare_data.onstart()

	app.run(host='0.0.0.0')
	#app.before_first_request( prepare_data.onstart()  )