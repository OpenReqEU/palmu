import sys
sys.path.append( "..")
import pytest
import requests
import json 
import os 
import json 
from  palmu import Palmu
import random

URL="http://localhost:9210/"

#URL = "http://217.172.12.199:9210/"
p = Palmu( refresh = True )
app = p.create_app()
test_client = app.test_client()

"""
def test_build_app():

	p = Palmu()
	app = p.create_app()
	test_client = app.test_client()
	assert app is not None
	shutdown_server()
"""

def test_build_app_refresh():

	assert app is not None



def test_palmu_related():

	#app = palmu.create_app()
	k = 10
	#params =  {'id': 'QDS-413' , "k" : str(k) , "m" : "1"}
	r = test_client.get("/getRelated?id=QDS-413&m=1&k=10"  )
	response = json.loads( r.data )
	print(response)
	print( r.data )
	assert k -1  == len( response["dependencies"] ) 
	#assert 1 == 1 


def test_update_reqs():

	p = Palmu()
	app = p.create_app()
	test_client = app.test_client()

	N = 2 
	counts = 0 
	files = os.listdir("./data/")
	files_json = [ "./data/"+f for f in files if ".json" in f ]
	for f in files_json[:N]:
		
		payload = None
		with open(f , "rb") as fhandler:
			payload = fhandler.read()

		payload = json.loads(payload)
		r = test_client.post( "/updateRequirements", json = payload )
		print( r.data )
		print( type(r.data) )
		response = r.get_json()
		if response["status"] == "ok":
			counts += 1 

	assert counts == N 

def test_find_by_new():

	p = Palmu()
	app = p.create_app()
	test_client = app.test_client()

	payload = None
	with open(  "./tests/new_issue.json" , "rb") as fhandler:
		payload = fhandler.read()
	payload = json.loads( payload )
	k = payload["k"]
	#assert k == 5 
	#payload["id"] = str( random.randint( 10 , 5000) )
	r = test_client.post( "/newIssue" , json = payload )
	
	response = r.get_json()
	assert  k-1 ==  len(response["dependencies"]) 

	payload["id"] = str( random.randint( 10 , 5000) )
	r = test_client.post( "/newIssue" , json = payload )
	
	response = r.get_json()
	assert  k-1 ==  len(response["dependencies"]) 

	#assert k -1  == len( r["dependencies"] ) 

"""
asdadas
""" 
