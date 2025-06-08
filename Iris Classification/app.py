import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from flask import Flask, request, render_template

# Create static folder if not exists
if not os.path.exists('static'):
    os.makedirs('static')

# Load data
df = pd.read_csv('Iris.csv')

# Save original species labels for plotting
df['Species_Label'] = df['Species']

# 1) Pairplot
sns.pairplot(df, hue="Species_Label")
plt.savefig('static/pairplot.png')
plt.close()

# Encode target labels
le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])

# 2) Correlation heatmap
plt.figure(figsize=(10,6))
numeric_df = df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('static/correlation_heatmap.png')
plt.close()

# Boxplots with original species labels
plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='Species_Label', y='SepalLengthCm')
plt.title('Boxplot of Sepal Length by Species')
plt.savefig('static/boxplot_sepal_length.png')
plt.close()

plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='Species_Label', y='SepalWidthCm')
plt.title('Boxplot of Sepal Width by Species')
plt.savefig('static/boxplot_sepal_width.png')
plt.close()

plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='Species_Label', y='PetalLengthCm')
plt.title('Boxplot of Petal Length by Species')
plt.savefig('static/boxplot_petal_length.png')
plt.close()

plt.figure(figsize=(10,6))
sns.boxplot(data=df, x='Species_Label', y='PetalWidthCm')
plt.title('Boxplot of Petal Width by Species')
plt.savefig('static/boxplot_petal_width.png')
plt.close()

# Prepare features and target
feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
x = df[feature_cols]
y = df['Species']

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Save scaler for later use
with open("static/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Hyperparameter tuning for KNN
param_grid = {
    "n_neighbors": [1,3,5,7,9,11],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"]
}
knn1 = KNeighborsClassifier()
grid_search = GridSearchCV(knn1, param_grid, cv=7, scoring="accuracy")
grid_search.fit(x_train, y_train)

# Train final KNN model with fixed n_neighbors=9 (or you can use grid_search.best_params_)
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(x_train, y_train)

# Predictions and evaluation
y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'KNN Accuracy with 9 Neighbors: {accuracy:.4f}')

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Setosa", "Versicolor", "Virginica"],
            yticklabels=["Setosa", "Versicolor", "Virginica"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("KNN (n_neighbors=9) Confusion Matrix")
plt.savefig('static/confusion_matrix.png')
plt.close()

# Save model and label encoder
with open("static/knn_model.pkl", "wb") as f:
    pickle.dump(knn, f)

with open("static/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("Model, scaler, and label encoder saved.")

# Load model, scaler, and encoder for prediction in Flask app
with open("static/knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)

with open("static/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("static/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Create Flask app
app = Flask(__name__)

# Pass accuracy as a global variable for easy access in routes
model_accuracy = accuracy  # float value like 0.9667

@app.route('/')
def home():
    images = [
        'pairplot.png',
        'correlation_heatmap.png',
        'confusion_matrix.png',
        'boxplot_sepal_length.png',
        'boxplot_sepal_width.png',
        'boxplot_petal_length.png',
        'boxplot_petal_width.png'
    ]
    return render_template('index.html', images=images, accuracy=model_accuracy)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        features_scaled = scaler.transform(features)
        prediction = knn_model.predict(features_scaled)
        species = label_encoder.inverse_transform(prediction)[0]

        # Map species to image filenames for species display (optional)
        species_to_image = {
            'setosa': 'setosa.jpg',
            'versicolor': 'versicolor.jpg',
            'virginica': 'virginica.jpg'
        }
        img_file = species_to_image.get(species.lower(), None)

        return render_template(
            "index.html",
            prediction_text=f'Predicted Species: {species}',
            image_file=img_file,
            images=[
                'pairplot.png',
                'correlation_heatmap.png',
                'confusion_matrix.png'
            ],
            accuracy=model_accuracy
        )
    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}",
            images=[
                'pairplot.png',
                'correlation_heatmap.png',
                'confusion_matrix.png'
            ],
            accuracy=model_accuracy
        )

if __name__ == '__main__':
    app.run(debug=True)
