from flask import Flask
from datetime import datetime
import re
from flask_restful import Resource, Api
from json import dumps
from flask_jsonpify import jsonify

app = Flask(__name__)

#@app.route("/")
#def home():
#    return "Hello, Flask!"

api = Api(app)

class CheckSpam(Resource):
    def get(self, text):

        result = {'Test':text}
        return jsonify(result)

api.add_resource(CheckSpam, '/CheckSpam/<text>') # Route_3w
