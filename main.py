from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS, cross_origin
from BTP import Functions_database as fd
import tensorflow as tf

from pickle import load

app = Flask(__name__)
api = Api(app)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

class StrengthData(Resource):
    def get(self):
        scaler = load(open('BTP/scaler.pkl', 'rb'))
        loaded_model = tf.keras.models.load_model('BTP/saved_model/my_model')

        args = request.args
        alloy = args.get("alloy").replace(" ","")
        condition = args.get("condition")
        param1_to_drop = args.get("param1_to_drop")
        param2_to_drop = args.get("param2_to_drop")
        params_to_drop = [param1_to_drop,param2_to_drop]

        print(params_to_drop)

        data = fd.easy_prediction(alloy, condition, scaler, loaded_model, params_to_drop)

        return {"data": f"{data} MPa"}
    
class AlloyList(Resource):
    def get(self):
        alloys = fd.get_alloys()
        return {"data" : alloys}

class ParamKeys(Resource):
    def get(self):
        params = fd.get_params()
        return {"data": params}
    
api.add_resource(StrengthData,"/strength_data")
api.add_resource(AlloyList,"/alloys")
api.add_resource(ParamKeys, "/params")

if __name__ == "__main__":
    print(app)
    print(api.resources)
    app.run(host='localhost', port=8000,debug=True)
