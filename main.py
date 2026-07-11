import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

DATA_DIR = "Data"
MODELS_DIR = "Models"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODELS_DIR, "model.pkl")
PIPELINE_FILE = os.path.join(MODELS_DIR, "pipeline.pkl")
HOUSING_CSV = os.path.join(DATA_DIR, "housing.csv")
INPUT_CSV = os.path.join(DATA_DIR, "input.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "output.csv")

def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())])
     
    cat_pipeline = Pipeline([
        ("onehot",OneHotEncoder(handle_unknown="ignore"))])
    
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    housing = pd.read_csv(HOUSING_CSV)

    housing["income_cat"] = pd.cut(housing["median_income"],bins=[0.,1.5,3.0,4.5,6.0,np.inf], labels=[1,2,3,4,5])

    split = StratifiedShuffleSplit(n_splits = 1,test_size = 0.2,random_state = 42)
    for train_index,test_index in split.split(housing,housing["income_cat"]):
        housing.loc[test_index].drop("income_cat", axis =1).to_csv(INPUT_CSV,index = False)
        housing = housing.loc[train_index].drop("income_cat", axis =1)
        
        

    

    housing_features = housing.drop("median_house_value", axis=1)
    housing_label = housing["median_house_value"].copy()

    num_attribs = housing_features.drop("ocean_proximity",axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    pipeline = build_pipeline(num_attribs,cat_attribs)

    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_label)

    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)

    print("Model trained and saved")

else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv(INPUT_CSV)
    features_only = input_data.drop("median_house_value", axis=1)
    transformed_input = pipeline.transform(features_only)
    predictions = model.predict(transformed_input)
    input_data["predicted_house_value"] = predictions
 
    input_data.to_csv(OUTPUT_CSV, index=False)
    print("Inference complete. Results saved to output.csv")


