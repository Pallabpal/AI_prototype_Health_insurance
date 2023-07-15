from flask import Flask, render_template, request, send_file
from sklearn.ensemble import RandomForestClassifier
from joblib import load, dump
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Set the upload and output folders
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

ss = load('standardscaler.joblib')
one_hot = load('one_hot_encoder.joblib')
pca = load('pca.joblib')
rf = load('random_forest_model.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'csv_file' not in request.files:
        return "No file found"

    csv_file = request.files['csv_file']
    csv_file.save(os.path.join(app.config['UPLOAD_FOLDER'], csv_file.filename))

    raw_df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], csv_file.filename))

    # Keep a copy of the original DataFrame with only the selected features
    input_df = raw_df.copy()

    input_df['Hospital Id'] = input_df['Hospital Id'].astype(np.int64)
    input_df['ccs_diagnosis_code'] = input_df['ccs_diagnosis_code'].astype(np.int64)
    input_df['ccs_procedure_code'] = input_df['ccs_procedure_code'].astype(np.int64)
    input_df['Code_illness'] = input_df['Code_illness'].astype(np.int64)
    input_df['Days_spend_hsptl'] = input_df['Days_spend_hsptl'].astype(np.float64)
    input_df['baby'] = input_df['baby'].astype(np.int64)

    mask_numeric = input_df.dtypes == float
    numeric_cols = input_df.columns[mask_numeric]
    numeric_cols = numeric_cols.tolist()
    input_df[numeric_cols] = ss.transform(input_df[numeric_cols])

    mask = input_df.dtypes == object
    object_cols = input_df.columns[mask]
    input_df = one_hot.transform(input_df)
    names = one_hot.get_feature_names_out()
    colunm_names = [name[name.find("_") + 1:] for name in [name[name.find("__") + 2:] for name in names]]
    input_df = input_df.toarray()
    input_df = pd.DataFrame(data=input_df, columns=colunm_names)
    input_df_hat = pca.transform(input_df)
    input_df_hat_PCA = pd.DataFrame(columns=[f'Projection  on Component {i + 1}' for i in range(len(input_df.columns))],
                                    data=input_df_hat)
    input_df_hat_PCA = input_df_hat_PCA.iloc[:, :4]

    predictions = rf.predict(input_df_hat_PCA)
    raw_df['predicted_class'] = predictions
    raw_df['predicted_class'] = raw_df['predicted_class'].apply(lambda x: 'Fraud' if x == 1 else 'Not Fraud')
    output_file = os.path.join(app.config['OUTPUT_FOLDER'], 'output.csv')
    raw_df.to_csv(output_file, index=False)

    return render_template('result.html', input_file=csv_file.filename, output_file='output.csv')
@app.route('/download_output')
def download_output():
    output_file = os.path.join(app.config['OUTPUT_FOLDER'], 'output.csv')
    return send_file(output_file, as_attachment=True)
@app.route('/download_text')
def download_text():
    text_file = 'text_file.txt'
    return send_file(text_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)



