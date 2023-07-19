## Libraries
import os
import random
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

figuresPath = os.path.join(os.getcwd(), "reports/figures/")
modelsPath = os.path.join(os.getcwd(), "models/")

def target_distribution(df,
                      targetCol,
                      uniqueTargets,
                      title,
                      labels):
  
  Count = df[targetCol].tolist()
  plt.figure(figsize=(6,6))
  plt.pie([Count.count(x)*100/len(Count) for x in uniqueTargets], 
          labels = labels,autopct='%1.1f%%')
  plt.title(title)

  # Saving figure by changing parameter values
  plt.savefig(f"{figuresPath}{targetCol}_dist.jpg", facecolor='w', bbox_inches="tight",
              pad_inches=0.3, transparent=True)
  print(f"{figuresPath}{targetCol}_dist.jpg saved successfully")


def plot_numeric_distribution(df,
                            col= "elevation"):
  df = df[[col, 'Label']]

  plt.figure(figsize=(15, 5))
  ax = sns.boxplot(data=df, x='Label', y=col)

  plt.xlabel('Target')
  plt.ylabel(col)
  plt.title(f'{col.upper()} Distribution')
  plt.xticks([0, 1], ['Normal', 'Landslide'])
  plt.grid(True)
  
  # Saving figure by changing parameter values
  plt.savefig(f"{figuresPath}{col}_dist.jpg", facecolor='w', bbox_inches="tight",
              pad_inches=0.3, transparent=True)
  print(f"{figuresPath}{col}_dist.jpg saved successfully")


## Seeder
# :seed to make all processes deterministic     # type: int
def seed_everything(seed=0):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)


def train_save_model(model,file_name, df, features):
  """
  DESCR
  -----
  This function trains and save a model

  Parameters
  ----------

  model: The model to be trained e.g Catboost, XGBoost, LightGBM etc.
  file_name: The name to save the model parameters with
  df: The data to be trained on.
  features: The selected features of the data to be trained on.
  """
  print("="*50)
  print(f"{file_name} Training...")
  print("="*50)

  model.fit(df[features],df.Label)

  pickle.dump(model, open(f'{modelsPath}{file_name}.pkl', 'wb'))

  print(f"\n{file_name} trained and saved successfully.")
  print("="*50)


def load_model(file_name, df, features):
  """
  DESCR
  -----
  This function trains and save a model

  Parameters
  ----------

  model: The model to be trained e.g Catboost, XGBoost, LightGBM etc.
  file_name: The name to save the model parameters with
  df: The data to be trained on.
  features: The selected features of the data to be trained on.
  """

  model = pickle.load(open(f'{modelsPath}{file_name}.pkl', 'rb'))
  prediction = model.predict(df[features])

  return prediction