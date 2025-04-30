from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression
import pandas as pd

app = Flask(__name__)

model = None
feature_columns = None

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with missing values
    df = df.dropna()

    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()

    return df


@app.route("/clean", methods=["POST"])
def clean():
    if 'file' not in request.files:
        return jsonify({"error": "CSV file must be uploaded with key 'file'"}), 400

    file = request.files['file']
    try:
        df = pd.read_csv(file, header=None, names=column_names)
        cleaned_df = clean_dataframe(df)
    except Exception as e:
        return jsonify({"error": f"Failed to clean data: {str(e)}"}), 400

    return jsonify(cleaned_df.to_dict(orient='records'))


@app.route("/train-cleaned", methods=["POST"])
def train_cleaned():
    global model, feature_columns

    data = request.get_json()
    df = pd.DataFrame(data)

    if 'MEDV' not in df.columns:
        return jsonify({"error": "'MEDV' column must be present as target"}), 400

    df = clean_dataframe(df)

    X = df.drop(columns=['MEDV'])
    y = df['MEDV']

    model = LinearRegression()
    model.fit(X, y)
    feature_columns = X.columns.tolist()

    return jsonify({
        "message": "Model trained successfully from cleaned data",
        "coefficients": model.coef_.tolist()
    })


@app.route("/predict-cleaned", methods=["POST"])
def predict_cleaned():
    global model, feature_columns

    if model is None:
        return jsonify({"error": "Model not trained yet"}), 400

    data = request.get_json()
    df = pd.DataFrame(data)

    df = clean_dataframe(df)

    # Ensure feature order
    X = df[feature_columns]

    preds = model.predict(X)
    return jsonify({"predictions": preds.tolist()})

if __name__ == "__main__":
    app.run()