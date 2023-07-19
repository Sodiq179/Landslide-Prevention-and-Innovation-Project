# Import libraries
import os
import utils
import pandas as pd

# Filter warnings
import warnings
warnings.filterwarnings('ignore')

## Input csv path
csv_path = os.path.join(os.getcwd(), "data/raw/")

# Import train csv
train = pd.read_csv(f"{csv_path}Train.csv")

# Exploratory Data Analysis
EDA = pd.DataFrame()
EDA['Label'] = train['Label']
EDA["geology_nunique"] = train[[x for x in train.columns if "geology" in x]].nunique(axis = 1)
for col in ['elevation', 'lsfactor', 'placurv', 'procurv', 'sdoif', 'slope', 'twi', 'aspect']:
  EDA[col] = train[[x for x in train.columns if col in x] ].mean(1)


# Visualizations
utils.target_distribution(df = EDA,
                  targetCol = 'Label',
                  uniqueTargets = [0, 1],
                  title = "Distribution of Landslides",
                  labels = ['Normal', 'Landslides'])


for column in ['elevation', 'lsfactor', 'sdoif', 'slope', 'twi', 'aspect']:
  utils.plot_numeric_distribution(df = EDA,
                              col= column)