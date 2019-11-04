# Import your application as:
# from app import application
# Example:
from palmu import Palmu
import json
# Import CherryPy
import os
if __name__ == '__main__':

    config = ""
    with open( "./config.json" , "r") as f:
        config = f.read()

    config = json.loads( config )
    ip = config["ip"]
    p = Palmu( refresh = False )
    app = p.create_app()
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    jsons_path = "./data"
    files = []
    files = os.listdir( "./data" )
    files_json = [ jsons_path+"/"+f for f in files if ".json" in f ]
    #
    print("Processing Json Files")
    print( files_json )
    #
    app.run(host=ip, port=9210 , extra_files = files_json , debug=True )


