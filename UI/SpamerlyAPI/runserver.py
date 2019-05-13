"""
This script runs the SpamerlyAPI application using a development server.
"""

from os import environ
from SpamerlyAPI import app

if __name__ == '__main__':
    HOST = environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(environ.get('SERVER_PORT', '7657'))
    except ValueError:
        PORT = 7657
    app.run(HOST, PORT)
