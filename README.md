ReadMe File
Project Title
Capstone Project: High-Risk Customer Classification for Online Purchases
Overview
This project aims to classify online purchase orders as high-risk or low-risk based on various attributes using machine learning models, specifically Gradient Boosting and Decision Tree Classifiers. The goal is to identify potential high-risk orders to help the business mitigate fraud risks effectively.
Requirements
To run the code provided in this project, you need the following dependencies installed:
* Python 3.x
* Libraries:
   * pandas
   * numpy
   * scikit-learn
   * matplotlib
   * seaborn
   * jupyter notebook (if you wish to use the Jupyter notebook)
You can install these libraries using the following command:
* pip install pandas numpy scikit-learn matplotlib seaborn
* pip install imbalanced-learn
Files Included
1. main.ipynb: A Jupyter notebook to run the classification model.
2. DataPreprocessing.py: Contains functions for data loading and preprocessing.
3. CustomerPredictionModel.py: Includes functions for training the model.
4. FeatureExtraction.py: Contains functions to extract the important features.
5. CostFunction.py: Contains functions to calculate the Misclassification cost.
6. ComputeMetrics.py: Contains functions to calculate accuracy, precision and recall scores.
7. README.md: This README file providing instructions for running the project.
8. risk-train.txt: The dataset used for training and testing the model.
________________


How to Run the Project
Step 1: Start the Application
Open the main.ipynb Jupyter notebook. This is the main file where the application starts running and orchestrates the various components of the project.
Step 2: Data Preprocessing
Inside main.ipynb, the first step is to import the DataPreprocessing class. This class handles loading and preprocessing the dataset, including cleaning the data and handling any missing values.
Step 3: Feature Extraction
Next, utilize the FeatureExtraction class to extract relevant features from the dataset. This step is crucial for enhancing the model's performance.
Step 4: Model Training and Prediction
1. After feature extraction, the CustomerPredictionModel class will be used to train the Gradient Boosting and Decision Tree models.
2. The models will be evaluated on both training and testing datasets.
Step 5: Cost Function and Metrics Calculation
1. Use the CostFunction class to calculate the total misclassification cost based on the model's predictions.
2. Finally, the ComputeMetrics class will be used to calculate metrics such as accuracy, precision, and recall to assess model performance.
Step 6: Visualization
The results, metrics, and visualizations will be displayed directly in the Jupyter notebook for easy reference and analysis.
