import os
import numpy as np
import pandas as pd

# Filter warnings
import warnings
warnings.filterwarnings('ignore')

groupCols = {"elevation":[],
             "lsfactor": [],
             "placurv": [],
             "procurv": [],
             "sdoif": [],
             "slope": [],
             "twi": [],
             "aspect": []
            }

for col in groupCols.keys():
  for i in range(1,22,5):
    groupCols[col].append([f"{x}_{col}" for x in range(i,i+5)])

#print(groupCols)

def build_features(df, train =True):
  """
  
  """
  temp = pd.DataFrame()
  selectedCols = ['elevation', 'lsfactor', 'placurv', 'procurv', 'sdoif', 'slope', 'twi', 'aspect']
  for i in selectedCols:
    temp[i+"_mean"] = df[[x for x in df.columns if i in x]].mean(axis = 1)
    temp[i+"_median"] = df[[x for x in df.columns if i in x]].median(axis = 1)
    temp[i+"_min"] = df[[x for x in df.columns if i in x]].min(axis = 1)
    temp[i+"_max"] = df[[x for x in df.columns if i in x]].max(axis = 1)
    temp[i+"_std"] = df[[x for x in df.columns if i in x]].std(axis = 1)
    temp[i+"_range"] = temp[i+"_max"] - temp[i+"_min"]
    temp[i+"_ratio1"] = temp[i+"_max"] / temp[i+"_min"]
    temp[i+"_ratio2"] = temp[i+"_min"] / temp[i+"_max"]

  for i in selectedCols:
    for idx in range(0,5):
      temp[i+f"_mean_5_{idx}"] = df[groupCols[i][idx]].mean(axis = 1)
      temp[i+f"_median_5_{idx}"] = df[groupCols[i][idx]].median(axis = 1)
      temp[i+f"_min_5_{idx}"] = df[groupCols[i][idx]].min(axis = 1)
      temp[i+f"_max_5_{idx}"] = df[groupCols[i][idx]].max(axis = 1)
      temp[i+f"_std_5_{idx}"] = df[groupCols[i][idx]].std(axis = 1)
      temp[i+f"_range_5_{idx}"] = temp[i+f"_max_5_{idx}"] - temp[i+f"_min_5_{idx}"]
      temp[i+f"_ratio1_5_{idx}"] = temp[i+f"_max_5_{idx}"] / temp[i+f"_min_5_{idx}"]
      temp[i+f"_ratio2_5_{idx}"] = temp[i+f"_min_5_{idx}"] / temp[i+f"_max_5_{idx}"]

  temp[[f"geology_{i}" for i in range(1, 8)]] = np.nan
  for index in range(len(temp)):
    current_dict = df[[x for x in df.columns if "geology" in x]].iloc[index,:].value_counts().to_dict()
    for i in range(1, 8):
      temp.loc[index, f"geology_{i}"] = current_dict.get(i, 0)

  temp["geology_nunique"] = df[[x for x in df.columns if "geology" in x]].nunique(axis = 1)

  if train:
    temp['Label'] = df.Label
    temp = pd.concat([temp, df.iloc[:, 1:-1]], axis = 1)
  else:
    temp = pd.concat([temp, df.iloc[:, 1:]], axis = 1)

  return temp



## Input csv path
csv_path = os.path.join(os.getcwd(), "data/raw/")
output_path = os.path.join(os.getcwd(), "data/processed/")

# Import train csv
train = pd.read_csv(f"{csv_path}Train.csv")
test = pd.read_csv(f"{csv_path}Test.csv")

new_train = build_features(train)
new_test = build_features(test, train=False)

new_train.to_csv(f"{output_path}new_train.csv", index=False)
new_test.to_csv(f"{output_path}new_test.csv", index=False)

print(f"Raw train shape : {train.shape}, Processed train shape : {new_train.shape}")
print(f"Raw test shape : {test.shape}, Processed test shape : {new_test.shape}")