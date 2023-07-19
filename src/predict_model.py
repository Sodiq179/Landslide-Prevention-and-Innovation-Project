# Import libraries
import os
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Other Libraries
from config import CFG
from utils import load_model, seed_everything

# Filter warnings
import warnings
warnings.filterwarnings('ignore')



## Input csv path
csv_path = os.path.join(os.getcwd(), "data/")
output_path = os.path.join(os.getcwd(), "data/output/")

# Import train csv
old_test = pd.read_csv(f"{csv_path}raw/Test.csv")
test = pd.read_csv(f"{csv_path}processed/new_test.csv")

# SEED
seed_everything(CFG.SEED)

# Catboost prediction 1
CATpred1 = load_model("catboost_model1",
          test,
          CFG.CATfeatures)

# Catboost prediction 2
CATpred2 = load_model("catboost_model2",
          test,
          CFG.CATfeatures)

# XGBoost prediction 1
XGBpred1 = load_model("xgboost_model1",
          test,
          CFG.XGBfeatures)

# XGBoost prediction 2
XGBpred2 = load_model("xgboost_model2",
          test,
          CFG.XGBfeatures)

# LGBoost prediction 1
LGBpred1 = load_model("lgboost_model1",
          test,
          CFG.LGBfeatures)

# LGBoost prediction 2
LGBpred2 = load_model("lgboost_model2",
          test,
          CFG.LGBfeatures)

blend = (CATpred1 + CATpred2 + XGBpred1 + XGBpred2 + LGBpred1 + LGBpred2)/6
preds = [1 if x >= 0.5 else 0 for x in blend]
sub_file = pd.DataFrame({'Sample_ID': old_test.Sample_ID, 'Label': preds})

# Submission file
sub_file.to_csv(f'{output_path}Submission.csv', index = False)
print(sub_file.head())