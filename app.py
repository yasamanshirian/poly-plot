import sqlite3
from flask import Flask, request
from flask_cors import CORS, cross_origin
import utils
import cv2


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/api/hello', methods=['GET', 'POST'])
@cross_origin()
def hello():
    poly = {}
    demo_image = ""
    anns = cv2.imread(demo_image)
    if request.method == 'GET':
        mask = utils.find_the_best_mask(point, anns)
        result = utils.ann_to_polygon_ui(mask, poly)
        #result = {'status': 1, 'message': 'test result for get'}
        return result
    elif request.method == 'POST':
        result = {'status': 1, 'message': 'test result for post'}
        return result

if __name__ == '__main__':
    app.debug = True
    app.run(debug=True,port=3000)
