from flask import Flask, request, jsonify
from flask_cors import CORS
from model import PropertyModel
import pandas as pd

app = Flask(__name__)
CORS(app)

model = PropertyModel("data.json")
model.train()

@app.route("/recommend", methods=["POST"])
def recommend():
    user_input = request.json
    df = model.df.copy()
    df['predicted_score'] = df.apply(lambda row: model.predict({
        'price': row['price'],
        'area': row['area'],
        'propertyType': row['propertyType'],
        'inventoryType': row['inventoryType'],
        'bhk': row['bhk'],
        'furnishing': row['furnishing'],
        'reraApproved': row['reraApproved'],
        'possession': row['possession'],
        'facing': row['facing']
    }), axis=1)

    top = df.sort_values(by='predicted_score', ascending=False).head(10)
    return jsonify(top[model.features + ['name', 'predicted_score']].to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
