import lime
import lime.lime_tabular
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
# Assuming X and y are already defined and preprocessed from your custom dataset


X = pd.read_csv('C:/Users/DELL/Desktop/Hackathons/Aeravat/Round 2/Code/streamlit/X.csv')
y = pd.read_csv('C:/Users/DELL/Desktop/Hackathons/Aeravat/Round 2/Code/streamlit/y.csv')
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train a RandomForestClassifier model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Initializing LIME Explainer
# Make sure feature_names is correctly defined. If X is a DataFrame, you can use X.columns.
# If X is a numpy array, you should manually define the feature names as a list.
feature_names = X.columns if hasattr(X, 'columns') else ['feature_' + str(i) for i in range(X.shape[1])]

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),  # Make sure training data is a numpy array
    feature_names=feature_names,
    class_names=np.unique(y),  # Assuming y contains the target class labels
    mode='classification'
)

def exp(instance,predict_fn):

    idx = 1  # Example index, adjust based on your dataset and indexing

    # Check if X_test is a DataFrame and use .iloc[idx] to access the instance
    # If it's a numpy array, access it with [idx]
    # instance = X_test.iloc[[idx]].values if hasattr(X_test, 'iloc') else X_test[idx]

    # # Ensure instance is formatted correctly for the model
    # # For pandas DataFrame, we used .values to get a numpy representation
    # # If it's a single instance, reshape might be necessary depending on the model input requirements
    # instance = instance.reshape(1, -1)

    # # It's crucial to pass a function that the explainer can use to make predictions
    # # This function should return probabilities
    # predict_fn = lambda x: model.predict_proba(x).astype(float)

    # Generate explanation
    explanation = explainer.explain_instance(instance[0], predict_fn, num_features=len(feature_names))


    # Extracting the feature names and their weights
    feature_importances = explanation.as_list()

    # This will give you a list of tuples where the first element is the feature and
    # the second element is the weight of that feature.
    for feature, weight in feature_importances:
        print(f"Feature: {feature}, Weight: {weight}")


    # Sorting features by their absolute importance
    sorted_features = sorted(feature_importances, key=lambda x: abs(x[1]), reverse=True)
    
    return explanation