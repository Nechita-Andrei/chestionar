import pandas as pd
from flask import Flask, request, jsonify
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

data = pd.read_csv("dog_breeds_extended.csv")

categorical_cols = ['size', 'allergy', 'trainability', 'energy', 'kids', 'grooming']

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), categorical_cols)],
    remainder='passthrough')

X = data.drop("breed", axis=1)
y = data["breed"]
X_encoded = preprocessor.fit_transform(X)

rf = RandomForestClassifier()
rf.fit(X_encoded, y)


@app.route('/predict', methods=['POST'])
def predict_breed():
    size = request.json['size']
    allergy = request.json['allergy']
    trainability = request.json['trainability']
    energy = request.json['energy']
    kids = request.json['kids']
    grooming = request.json['grooming']

    user_data = [[size, allergy, trainability, energy, kids, grooming]]
    user_data_encoded = preprocessor.transform(pd.DataFrame(user_data, columns=categorical_cols))
    breed = rf.predict(user_data_encoded)

    response = {'breed': breed[0]}
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=False)
