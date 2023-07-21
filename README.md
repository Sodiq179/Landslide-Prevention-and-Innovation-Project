# Landslide-Prevention-and-Innovation

<p align="center">
    <img width="400" src=reports/figures/Landslideimage.png alt="Land slide">
</p>


## Brief Description: 

Hong Kong, one of the hilliest and most densely populated cities in the world, is frequently hit by extreme rainfall and is therefore highly susceptible to rain-induced landslides. A landslide is the movement of masses of rock, debris, or earth down a slope and can result in significant loss of life and property. A high-quality landslide inventory is essential not only for landslide hazard and risk analysis but also for supporting agency decisions on landslide hazard mitigation and prevention.

The common practice of identifying landslides is visual interpretation which, however, is labor-intensive and time-consuming. Thus, this project will focus on automating the landslide identification process using artificial intelligence techniques, and target at using high-resolution terrain information to perform the terrain-based landslide identification.. The challenge is part of the [Zindi Competetion](https://zindi.africa/competitions/landslide-prevention-and-innovation-challenge).

## Project Organization
-----------------------

    ├── LICENSE
    ├── README.md        
    ├── data
    │   ├── raw
    │   │   ├── Test.csv               <- Downloaded test data from Zindi
    │   │   ├── Train.csv              <- Downloaded train data from Zindi
    │   │   └── SampleSubmission.csv   <- Downloaded submission data from Zindi
    │   ├── processed
    │   │   ├── new_test.csv          <- Generated test data [after running src/build_features.py]
    │   │   └── new_train.csv         <- Generated train data [after running src/build_features.py]
    │   └── output
    │       └── Submission.csv        <- Generated submission data [after running src/train_model.py and src/predict_model.py]
    ├── models                        <- Contains the saved XGBoost, CatBoost and LightBoost
    ├── notebooks
    │   ├── Analysis_notebook.ipynb               <- Analysis Notebook
    │   ├── scripts_runner_notebook.ipynb         <- Notebook for running the scripts (using GPU) 
    │   └── complete_model_implementation.ipynb   <- Notebook for complete implementation of the project (From EDA to submission) 
    ├── reports            
    │   └── figures      <- Generated graphics and figures [after running src/visualize.py]
    ├── requirements.txt        <- Requirements text file
    ├── src                     <- Source code for use in this project.
    │   ├── __init__.py         <- Makes src a Python module
    │   ├── build_features.py   <- Script for feature engineering.
    │   ├── predict_model.py    <- Script to mske prediction snd creste thr submission file.
    │   ├── train_model.py      <- Script to train the models (XGBoost, CatBoost and LGBoost).
    │   └── visualize.py        <- Script to generate the analysis graphics.
    └── models 

### How to run the codes

### Step 1: Donwload Script Runner Notebook

Download the `scripts_runner_notebook.ipynb` [here](notebooks/scripts_runner_notebook.ipynb) and upload it to your google colab.

### Step 2: Enable GPU on colaboratory

<p align="center">
    <img width="200" src=reports/figures/colab_gpu_activator.PNG alt="Activate GPU">
</p>

### Step 2: SETUP
**Setup by running the following codes in the `scripts_runner_notebook.ipynb` notebook**

**Connect colab to your google drive**

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Change working directory to your google drive**

```bash
%cd drive/MyDrive
```

**Fork the repository 🍴**

```bash
!git clone https://github.com/Sodiq179/Landslide-Prevention-and-Innovation-Project
```

**Change working directory to the project folder**

```bash
%cd "Landslide-Prevention-and-Innovation-Project"
```

**Install `requirement.txt` packages**
```bash
%%capture
!pip install -r requirement.txt
```

**Run the `visualize.py` script to create necessary visualizations**
```bash
!python src/visualize.py
```

**Run the `build_features.py` script to create new features**
```bash
!python src/build_features.py
```

**Run the `train_model.py` script to train the models**
```bash
!python src/train_model.py
```

**Run the `predict_model.py` script to make predictions**
```bash
!python src/predict_model.py
```


## [Rank on the Leaderboard](https://zindi.africa/competitions/landslide-prevention-and-innovation-challenge/leaderboard)

**Rank : 1/173**  

<p align="center">
    <img width="400" src=reports/figures/leaderboard.PNG alt="Leaderboard">
</p>


## Author

<div align='center'>

| Name           |                     Zindi ID                     |
|----------------|--------------------------------------------------|
|Sodiq Babawale |[@Babawale_Sodiq](https://zindi.africa/users/Babawale_sodiq)|


</div>

Don't forget to Leave a⭐️ to my repo, if you find it useful😊
