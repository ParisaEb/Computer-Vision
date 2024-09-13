# Iris Dataset Analysis and Model Training
Overview
This project involves analyzing the famous Iris dataset using various Python libraries. The analysis includes visualizing the data with pair plots, splitting the data into training and testing sets, and training different machine learning models. The models used include a Random Forest Classifier and a Multilayer Perceptron (MLP) neural network.

Exercises
Exercise 2: Iris Dataset Visualization and Random Forest Classifier
Load the Iris dataset:

The dataset is loaded using the sklearn.datasets.load_iris() method.
Data Visualization:

A pair plot of the Iris dataset is created using Seaborn to visualize the relationships between the features. This helps in understanding the distribution of the data across different species.
Data Splitting:

The data is split into training and testing sets using an 80/20 ratio. The first four columns are used as input features, and the last column is the target variable.
Model Training:

A RandomForestClassifier is trained on the training data and then used to make predictions on the test data.
The model's performance is evaluated using accuracy, a confusion matrix, and a classification report.
Exercise 3: Perceptron Model
Load the Iris dataset:

Similar to Exercise 2, the dataset is loaded using the sklearn.datasets.load_iris() method.
Data Preprocessing:

The data is standardized using StandardScaler to ensure the features have a mean of 0 and a standard deviation of 1.
Model Training:

A perceptron model with a hidden layer of 16 neurons is created using MLPClassifier.
The model is trained on the scaled training data and evaluated on the test data.
Exercise 4: Multilayer Perceptron on MNIST Dataset
Load MNIST Dataset:

The MNIST dataset is loaded using the keras.datasets.mnist module.
Data Preprocessing:

The data is flattened, normalized, and one-hot encoded to prepare it for neural network training.
Model Training:

A Multilayer Perceptron (MLP) is created and trained on the MNIST dataset.
The model is evaluated on the test data, and its accuracy is reported.
Getting Started
Prerequisites
Python 3.x
Jupyter Notebook or Google Colab
Required Libraries:
numpy
pandas
matplotlib
seaborn
sklearn
keras
Installation
Install the required Python libraries using pip:


pip install numpy pandas matplotlib seaborn scikit-learn keras
# Running the Code
You can run the code in a Jupyter Notebook or directly in Google Colab. To run in Colab, simply click the "Open in Colab" button at the top of this README.

Results
Perceptron Model: Achieved an accuracy of 93.33% on the Iris dataset test set.
Multilayer Perceptron on MNIST: Achieved an accuracy of approximately 98% on the MNIST test set.
License
This project is licensed under the MIT License.
