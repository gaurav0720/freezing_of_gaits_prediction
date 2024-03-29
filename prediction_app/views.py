from django.shortcuts import render
import numpy as np
import joblib
import os

# Get the path to the models directory
models_dir = os.path.join(os.path.dirname(__file__), 'models')

# Load the scaler and model
scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
model = joblib.load(os.path.join(models_dir, 'random_forest_model.pkl'))

# Function to scale input data using the loaded scaler
def scale_data(data, scaler):
    return scaler.transform(data.reshape(1, -1))

# Function to make predictions using the loaded model
def predict(data, model):
    return model.predict(data)

def index(request):
    if request.method == 'POST':
        input_str = request.POST.get('input_data')
        input_list = [float(x.strip()) for x in input_str.split(',')]
        input_data = np.array(input_list)
        scaled_data = scale_data(input_data, scaler)
        prediction = predict(scaled_data, model)
        if prediction[0] == 1:
            result = "No Freezing of Gaits"
        else:
            result = "Freezing of Gaits"
        return render(request, 'index.html', {'result': result})
    return render(request, 'index.html')
