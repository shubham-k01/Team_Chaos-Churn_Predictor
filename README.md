# Team_Chaos-Churn_Predictor
Hackathon project of Aeravat 1.0 at SPIT

This project predicts customer churn of a telecom company, which is the likelihood of a customer discontinuing their service or subscription. By identifying customers at risk of churning, businesses can take proactive steps to retain them.

Getting Started

This project requires Python libraries commonly used for data science tasks. Refer to the requirements.txt file for a list of dependencies and install them using 
> pip install -r requirements.txt.

Project Structure

data: This folder contains the raw and processed customer data.
notebooks: Jupyter notebooks for data exploration, feature engineering, model training, and evaluation.
models: This folder stores the trained machine learning models for churn prediction.
scripts: Any helper scripts used in the project reside here.
requirements.txt: This file lists the required Python libraries.
README.md: This file (you are reading it now!).
Project Overview

The notebooks within the notebooks folder guide you through the following steps:

Data Loading and Exploration: Load the customer data, explore its characteristics, and identify potential issues.
Data Cleaning and Preprocessing: Clean the data by handling missing values, outliers, and encoding categorical features.
Feature Engineering: Create new features that might be useful for churn prediction.
Model Training and Evaluation: Train various machine learning models to predict customer churn. Evaluate the performance of each model using metrics like accuracy, precision, recall, and F1 score.
Model Selection: Choose the best performing model based on the evaluation results.
Running the Notebooks

Clone this repository to your local machine.
Open a terminal and navigate to the project directory.
Start a Jupyter Notebook server using jupyter notebook.
Open the notebooks in the notebooks folder and run the code cells sequentially.
Further Exploration

Try implementing different machine learning algorithms for churn prediction.
Explore techniques for model deployment and serving predictions in real-time.
Analyze the factors that contribute most to customer churn based on the chosen model.
This project provides a starting point for building a customer churn prediction system. Feel free to adapt and expand upon it based on your specific needs and data.
