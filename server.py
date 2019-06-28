# Import your application as:
# from app import application
# Example:

from palmu import app

# Import CherryPy
import cherrypy
import os 
if __name__ == '__main__':

	jsons_path = "./data"
	files = []
	files = os.listdir( "./data" )
	files_json = [ jsons_path+"/"+f for f in files if ".json" in f ]
	print("Processing Json Files")
	print( files_json )

	app.run(host='0.0.0.0' , extra_files = files_json , debug=True )



	"""
    # Mount the application
    cherrypy.tree.graft(app, "/")

    # Unsubscribe the default server
    cherrypy.server.unsubscribe()

    # Instantiate a new server object
    server = cherrypy._cpserver.Server()

    # Configure the server object
    server.socket_host = "0.0.0.0"
    server.socket_port = 9210
    server.thread_pool = 30

    # For SSL Support
    # server.ssl_module            = 'pyopenssl'
    # server.ssl_certificate       = 'ssl/certificate.crt'
    # server.ssl_private_key       = 'ssl/private.key'
    # server.ssl_certificate_chain = 'ssl/bundle.crt'

    # Subscribe this server
    server.subscribe()

    # Start the server engine (Option 1 *and* 2)

    cherrypy.engine.start()
    cherrypy.engine.block()
	"""
