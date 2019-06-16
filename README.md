# Grab-safety challenge submission
Contact : boraeang [at] hotmail.com
# About the submission
The solution has been developed using an AWS EC2 instance (r4.4xlarge) with the AMI ID
Deep Learning AMI (Amazon Linux) Version 23.0 <br>
In order to make a prediction make sure that the "data" folder follows the below structure:<br>

    .
    ├── data                        # Data folder
    |   ├── 0-raw_data              # Folder containing all the raw data in .csv format
    |   |   ├── safety              # Folder containing the original data for the challenge
    |   |   │   ├── features        # Original features data
    |   |   |   └── labels          # Original labels data
    |   |   └── holdout             # Folder containing the hold-out data for the challenge
    |   |       ├── features        # Hold-out features data
    |   |       └── labels          # Hold-out labels data
    |   ├── 1-preprocessed          # Folder containing the preprocessed data in .parquet format
    |   ├── 2-features              # Folder containing the extracted features with labels in .parquet format
    |   ├── 3-ml_datasets           # Folder containing the split between training set, validation set and hold-out set
    |   └── 4-results               # Final prediction of the model (on validation set and hold-out set)
    |
    ├── 00_preprocessing            # Notebooks folder to process the data
    ├── 01_training                 # Notebook folder to create the model (No need to re-execute it)
    └── 02_prediction               # Notebook folder to validate the performance of the model either on validation or hold-out set
    
    
    
    
>

# To run a prediction on holdout data:
1 - Clone this project in your home folder<br>
2 - Put the hold-out files in the folder ~/grab-safety/data/0-raw_data/holdout/<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"features" files of the hold-out data must be in "/grab-safety/data/0-raw_data/holdout/features/"<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"labels" unique file of the hold-out data must be in "/grab-safety/data/0-raw_data/holdout/labels/"<br>
3 - Execute the python notebook "05-preprocess-holdout-data.ipynb" located in the "/grab-safety/00_preprocessing/" folder<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;* Becareful extracting features with tsfresh takes approx. 4 hours even with an EC2 r4.4xlarge instance<br>
4 - Once the features are created, go to the "/grab-safety/02_prediction/" folder and execute the python notebook "01-apply-lgb-model.ipynb"<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The lightgbm model classifier has already been trained and is located in the folder "/grab-safety/02_prediction/models/"<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the notebook will return the performance of the model on the hold-out data<br>

# To retrain the model (not necessary as the model has been saved in this project):
1 - Go to the "/grab-safety/00_preprocessing/" folder and run all the python notebooks in their respective order<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;01-preprocess-data.ipynb<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;02a-feature-engineering-xgboost.ipynb<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;02b-feature-engineering-xgboost2.ipynb<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;03_split-data.ipynb<br>
2 - Go to the "/grab-safety/01_training/" folder and run the 01-create-lgb-model.ipynb notebook<br>


<img src="https://static.wixstatic.com/media/397bed_e0fd4340ff5f40de876b26f0fb7e1f83~mv2.png/v1/fill/w_610,h_610,al_c,q_85,usm_0.66_1.00_0.01/Grab%20EDM_Safety.webp"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />
