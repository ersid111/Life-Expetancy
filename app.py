from flask import Flask,render_template,request
import numpy as np
import pickle
import json
model = pickle.load(open("model.pkl","rb"))

app = Flask(__name__)
@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    data = request.form
    input_data = np.zeros(19)
   
    input_data[0] = (request.form['Year'])
    input_data[1] = (data['Adult Mortality'])
    input_data[2] = (data['infant deaths'])
    input_data[3] = data['Alcohol']
    input_data[4] = data['percentage expenditure']
    input_data[5] = data['Hepatitis B']
    input_data[6] = data['Measles ']
    input_data[7] = request.form['bmi']
    input_data[8] = data['under-five deaths ']
    input_data[9] = data['Polio']
    input_data[10] = data['Total expenditure']
    input_data[11] = data['Diphtheria ']
    input_data[12] = data[' HIV/AIDS']
    input_data[13] = data['GDP']
    input_data[14] = data[' thinness  1-19 years']
    input_data[15] = data[' thinness 5-9 years']
    input_data[16] = data['Income composition of resources']
    input_data[17] = data['Schooling']
    input_data[18] = np.log(int(data['log_population']))
    print(input_data)
    result = model.predict([input_data])
   
    return render_template('index.html',prediction=result)

if __name__== "__main__":
    app.run(debug=True)



