from flask import Flask, render_template, request , send_from_directory

app = Flask(__name__)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np

data = pd.read_csv(r'D:\Project\Diabetic - Sheet1 (1).csv')
X = data[['Fasting Glucose', 'PP ']]
y = data['output']
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = SVC(kernel='rbf', C=1.0, random_state=42)
clf.fit(X_train, y_train)

# Create a route to handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data1 = float(request.form['Fasting Glucose'])
        input_data2 = float(request.form['PP '])
        input_data = (input_data1, input_data2)
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        std_data = scaler.transform(input_data_reshaped)
        prediction = clf.predict(std_data)
        if prediction[0] == 0:
            result = 'Pre Diabetic Patient'
        elif prediction[0] == -1:
            result = 'Non-Diabetic Patient'
        else:
            result = 'Diabetic Patient'
        return render_template('result.html', result=result)

# Create a route to render the input form
@app.route('/')
def index():
    return render_template('index3.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True)
