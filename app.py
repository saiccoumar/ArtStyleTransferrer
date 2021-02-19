
from flask import Response
from flask import Flask
from flask import render_template
import os

current_directory = os.getcwd()
print(current_directory)
statDir = current_directory+'/static/'
templateDir = current_directory+'/templates/'
# initialize a flask object
app = Flask(__name__,static_folder=statDir,
            template_folder=templateDir)



@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	app.run(port=8000, debug=True, use_reloader=False)
