from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

# Load data from CSV file
data = pd.read_csv("dog_breeds_extended.csv")

# Identify categorical columns
categorical_cols = ['size', 'allergy', 'trainability', 'energy', 'kids', 'grooming']

# Encode categorical columns using OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), categorical_cols)],
    remainder='passthrough')

# Fit the preprocessor and transform the data
X = data.drop("breed", axis=1)
y = data["breed"]
X_encoded = preprocessor.fit_transform(X)

# Train a random forest classifier on the encoded data
rf = RandomForestClassifier()
rf.fit(X_encoded, y)


# Define a function to predict the dog breed based on user input
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
    accuracy_scores = cross_val_score(rf, X_encoded, y, cv=5)  # Change cv value as desired

    # Calculate the mean accuracy score
    mean_accuracy = accuracy_scores.mean()

    print("Accuracy:", mean_accuracy)

    response = {'breed': breed[0]}
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=False)
