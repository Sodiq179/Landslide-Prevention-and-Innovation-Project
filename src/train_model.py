# Import libraries
import os
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Other Libraries
from config import CFG
from utils import train_save_model, seed_everything

# Filter warnings
import warnings
warnings.filterwarnings('ignore')



## Input csv path
csv_path = os.path.join(os.getcwd(), "data/processed/")

# Import train csv
train = pd.read_csv(f"{csv_path}new_train.csv")

# SEED
seed_everything(CFG.SEED)

# Catboost model 1
train_save_model(CatBoostClassifier(**CFG.cat_params_1),
                "catboost_model1",
                train,
                CFG.CATfeatures)

# Catboost model 2
train_save_model(CatBoostClassifier(**CFG.cat_params_2),
                "catboost_model2",
                train,
                CFG.CATfeatures)

# XGBoost model 1
train_save_model(xgb.XGBClassifier(**CFG.xgb_params_1),
                "xgboost_model1",
                train,
                CFG.XGBfeatures)

# XGBoost model 2
train_save_model(xgb.XGBClassifier(**CFG.xgb_params_2),
                "xgboost_model2",
                train,
                CFG.XGBfeatures)

# LGBoost model 1
train_save_model(lgb.LGBMClassifier(**CFG.lgb_params_1),
                "lgboost_model1",
                train,
                CFG.LGBfeatures)

# LGBoost model 2
train_save_model(lgb.LGBMClassifier(**CFG.lgb_params_2),
                "lgboost_model2",
                train,
                CFG.LGBfeatures)
