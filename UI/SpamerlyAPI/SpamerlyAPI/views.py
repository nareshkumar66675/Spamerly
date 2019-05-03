"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template
from SpamerlyAPI import app
from flask import Flask, request
from flask_restful import Resource, Api
from json import dumps
from flask.ext.jsonpify import jsonify
#@app.route('/')
#@app.route('/home')
#def home():
#    """Renders the home page."""
#    return render_template(
#        'index.html',
#        title='Home Page',
#        year=datetime.now().year,
#    )

#@app.route('/contact')
#def contact():
#    """Renders the contact page."""
#    return render_template(
#        'contact.html',
#        title='Contact',
#        year=datetime.now().year,
#        message='Your contact page.'
#    )

#@app.route('/about')
#def about():
#    """Renders the about page."""
#    return render_template(
#        'about.html',
#        title='About',
#        year=datetime.now().year,
#        message='Your application description page.'
#    )
api = Api(app)

class CheckSpam(Resource):
    def get(self, text):

        result = {'Test':text}
        return jsonify(result)

api.add_resource(CheckSpam, '/CheckSpam/<text>') # Route_3w