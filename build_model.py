import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    from future_encoders import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

DATASETS_PATH = "lib/data/"
MODELS_PATH = "lib/models/"

def load_grading_data(datasetfile):
    filepath = DATASETS_PATH + datasetfile
    return pd.read_csv(filepath)

def build_model():

    # Get csv file as pandas dataframe
    grading = load_grading_data("student-mat.csv")
    grading_with_id = grading.reset_index()
    
    # Split the Dataframe into training and testing sets randomly
    train_set, test_set = train_test_split(grading_with_id, test_size=0.2, random_state=42)

    # Separate features from labels in training set
    training_features = train_set.drop("G3", axis=1)
    training_labels = train_set["G3"].copy()

    # Get a dataframe with only numerical attributes to get a list of its attributes for the pipeline later
    training_num = training_features.drop("school", axis=1)
    training_num = training_num.drop("sex", axis=1)
    training_num = training_num.drop("address", axis=1)
    training_num = training_num.drop("famsize", axis=1)
    training_num = training_num.drop("Pstatus", axis=1)
    training_num = training_num.drop("Mjob", axis=1)
    training_num = training_num.drop("Fjob", axis=1)
    training_num = training_num.drop("reason", axis=1)
    training_num = training_num.drop("guardian", axis=1)
    training_num = training_num.drop("schoolsup", axis=1)
    training_num = training_num.drop("famsup", axis=1)
    training_num = training_num.drop("paid", axis=1)
    training_num = training_num.drop("activities", axis=1)
    training_num = training_num.drop("nursery", axis=1)
    training_num = training_num.drop("higher", axis=1)
    training_num = training_num.drop("internet", axis=1)
    training_num = training_num.drop("romantic", axis=1)

    # Create a pipeline to standardize the numerical features and  bring all the numeric values to the same scale (Standardization is (x-mean)/variance)
    num_pipeline = Pipeline([
        ("std_scaler", StandardScaler()),
    ])

    # Create a full pipeline to standardize numerical attributes and one hot encode categorical attributes
    num_attributes = list(training_num)
    cat_attributes = ["school", "sex", "address", "famsize", "Pstatus", "Mjob", "Fjob", "reason", "guardian", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet",	"romantic"]
    
    full_pipeline = ColumnTransformer([
        ("num_pipeline", num_pipeline, num_attributes),
        ("cat_pipeline", OneHotEncoder(), cat_attributes),
    ])

    # Pass the Dataframe through the full pipeline
    training_prepared = full_pipeline.fit_transform(training_features)

    # Use a regression model called RandomForestRegressor
    forest_reg = RandomForestRegressor()
    forest_reg.fit(training_prepared, training_labels)

    print("Model training complete: RandomForestRegressor")

    # Show validity of RandomForestRegressor using K-Fold cross-validation
    neg_mse_scores = cross_val_score(forest_reg, training_prepared, training_labels, scoring="neg_mean_squared_error", cv=10)
    rmse_forest_scores = np.sqrt(-neg_mse_scores)
    print("Mean of cross validated RMSE for RandomForestRegressor:", rmse_forest_scores.mean())
    print("Std Dev of cross validated RMSE for RandomForestRegressor:", rmse_forest_scores.std())
    
    # Try our the random_forest_regressor on testing model
    # Separate the features from the labels in the test set
    testing_features = test_set.drop("G3", axis=1)
    testing_labels = test_set["G3"].copy()

    # Pass the testing features through the pipeline
    testing_prepared = full_pipeline.transform(testing_features)

    # Predict on testing data
    test_predictions = forest_reg.predict(testing_prepared)

    # Print our testing RMSE
    test_mse = mean_squared_error(testing_labels, test_predictions)
    test_rmse = np.sqrt(test_mse)
    print("Testing RMSE: ", test_rmse)

    # Save the model using Python pickle module
    model_pickle_file_path = MODELS_PATH + "grading_random_forest_regressor_model.pkl" 
    joblib.dump(forest_reg, model_pickle_file_path)
    print("Saved Model at: ", model_pickle_file_path)

if __name__ == "__main__":
    build_model()

