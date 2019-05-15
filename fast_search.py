import numpy as np

import pandas as pandas

import faiss 
import pickle 
import argparse

import prepare_data
import os

import json
from flask import Flask
from flask import request

from dataManager import DataManager
app = Flask(__name__)

parser = argparse.ArgumentParser(description=' Get similar reports from jira data')

parser.add_argument('--id', type =str , required = True )

EMB_FILE = "./data/embbedings200.npy"
MAP_FILE = "./data/mappings200.map"
GLOVE_FILE = "./data/glove.6B.200d.txt" 

dm = DataManager( GLOVE_FILE  )
"""
def load_data(  emb_file , map_file ):
	
	emb = np.load( emb_file )

	mapping  = pickle.load( open(map_file , "rb") )

	return emb, mapping

def get_index():

	embs , mapping = load_data( EMB_FILE , MAP_FILE )

	embs = embs.astype( np.float32 )
	#D = embs.shape[1]
	D = embs.shape[1]

	#print( embs.shape )
	#print(D)
	index = faiss.IndexFlatL2( D )
	#print( index.is_trained )
	index.add( embs )
	#print( embs.shape )
	k = 3 
	D, I = index.search( embs [:5], k) 

	return index ,  embs , mapping

def search( qtid , index   ,  embs , mappings , k = 4  ):
    if not qtid in mappings:
        return None

    ind = mappings[ qtid ]
    # got the vector

    vector = embs[ ind , : ].reshape( (1 , embs.shape[1] ))
    print( vector.shape )
    distances , I = index.search( vector , k )

    return I[0]
"""
@app.route("/getRelated", methods=['GET'])
def main():

    idd = request.args.get('id')


    print( "Query issue: " , idd )
    similar_issues = dm.find_by_id( idd , k = 10 ) 
    if similar_issues  == []:
        return "No such ID found"

    return json.dumps(similar_issues)

@app.route("/postProject", methods=['POST'])
def post_project():

    data = request.get_json()
    if data is None:
        return 'No data posted!'

    project_name = data["projects"][0]["id"]

    print("New project: ", project_name)

    filename = project_name + '.json'

    path = os.path.join('./data/', filename)

    print(path)

    with open(path, 'w') as json_file:
        json.dump(data, json_file)
        json_file.close()

    prepare_data.process_files()

    return "Project added"


#app.before_first_request( prepare_data.onstart() )

if __name__ == '__main__':

	#prepare_data.onstart()

	app.run(host='0.0.0.0')
	#app.before_first_request( prepare_data.onstart()  )
