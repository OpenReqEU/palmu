# Import your application as:
# from app import application
# Example:

#from palmu import app
from palmu import Palmu
# Import CherryPy
import os 
if __name__ == '__main__':

	p = Palmu( refresh = True )
	app = p.create_app()
	os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
	jsons_path = "./data"
	files = []
	files = os.listdir( "./data" )
	files_json = [ jsons_path+"/"+f for f in files if ".json" in f ]
	#files_json.append("./data/nohay.no")
	print("Processing Json Files")
	print( files_json )	
	#files_json = []
	app.run(host='0.0.0.0', port=9210 , extra_files = files_json , debug=True )

	#assert  4 ==  len(r["dependencies"]) 