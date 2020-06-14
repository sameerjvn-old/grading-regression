import requests

url = 'http://127.0.0.1:5000/predictGrade'
# params ={'query': 'that movie was boring'}
params ={'school': 'GP',
         'sex': 'F',
         'age': 18,
         'address': 'U',
         'famsize': 'GT3',
         'Pstatus': 'A',
         'Medu': 4,
         'Fedu': 4,
         'Mjob': 'at_home',
         'Fjob': 'teacher',
         'reason': 'course',
         'guardian': 'mother',
         'traveltime': 2,
         'studytime': 2,
         'failures': 0,
         'schoolsup': 'yes',
         'famsup': 'no',
         'paid': 'no',
         'activities': 'no',
         'nursery': 'yes',
         'higher': 'yes',
         'internet': 'no',
         'romantic': 'no',
         'famrel': 4,
         'freetime': 3,
         'goout': 4,
         'Dalc': 1,
         'Walc': 1,
         'health': 3,
         'absences': 6,
         'G1': '5',
         'G2': '6'}
#params = {'row_1': ["GP","F",18,"U","GT3","A",4,4,"at_home","teacher","course","mother",2,2,0,"yes","no","no","no","yes","yes","no","no",4,3,4,1,1,3,6,"5","6"]}
response = requests.get(url, params)
response.json()