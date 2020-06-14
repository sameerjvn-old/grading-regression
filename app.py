from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
import pandas as pd
from sklearn.externals import joblib

app = Flask(__name__)
api = Api(app)

# Load the model object
MODELS_PATH = "lib/models/"
file_name = "grading_random_forest_regressor_model.pkl"
file_path = MODELS_PATH + file_name
forest_reg = joblib.load(file_path)

# Parse the argument
parser = reqparse.RequestParser()
parser.add_argument('school')
parser.add_argument('sex')
parser.add_argument('age')
parser.add_argument('address')
parser.add_argument('famsize')
parser.add_argument('Pstatus')
parser.add_argument('Medu')
parser.add_argument('Fedu')
parser.add_argument('Mjob')
parser.add_argument('Fjob')
parser.add_argument('reason')
parser.add_argument('guardian')
parser.add_argument('traveltime')
parser.add_argument('studytime')
parser.add_argument('failures')
parser.add_argument('schoolsup')
parser.add_argument('famsup')
parser.add_argument('paid')
parser.add_argument('activities')
parser.add_argument('nursery')
parser.add_argument('higher')
parser.add_argument('internet')
parser.add_argument('romantic')
parser.add_argument('famrel')
parser.add_argument('freetime')
parser.add_argument('goout')
parser.add_argument('Dalc')
parser.add_argument('Walc')
parser.add_argument('health')
parser.add_argument('absences')
parser.add_argument('G1')
parser.add_argument('G2')

# Define the Resource class object
class PredictGrade(Resource):
    def get(self):
        # Use parser and find user's queries
        args = parser.parse_args()
        print("Hello, Arguments: ", args)
        
        # Convert JSON to dataframe and rearrange the columns
        data = pd.DataFrame.from_dict(args, orient='index')
        data = data.transpose()
        data = data[['school', 'sex', 'age','address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2']]
        print(data)
        print(data.shape)
        print(data.info())
        
        # Change string values in predicting row back to numbers
        
        # Vectorize the arguments and predict
        
        # Predict the grade
        
        # Return output
        return
    
# Setup API endpoint
api.add_resource(PredictGrade, '/predictGrade')
    
if __name__ == '__main__':
    app.run(debug=True)