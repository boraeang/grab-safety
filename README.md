# Grab-safety challenge submission

# To run a prediction on holdout data:
1 - Clone this project in your home folder<br>
2 - Put the hold-out files in the folder ~/grab-safety/data/0-raw_data/holdout/<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"features" files of the hold-out data must be in "~/grab-safety/data/0-raw_data/holdout/features/"<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"labels" unique file of the hold-out data must be in "~/grab-safety/data/0-raw_data/holdout/labels/"<br>
3 - Execute the python notebook "05-preprocess-holdout-data.ipynb" located in the "~/grab-safety/00_preprocessing/" folder<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Becareful extracting features with tsfresh takes a long time even with an EC2 r4.4xlarge instance<br>
4 - Once the features are created, go to the "~/grab-safety/02_prediction/" folder and execute the python notebook "01-apply-lgb-model.ipynb"<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The lightgbm model classifier has already been trained and is located in the folder "~/grab-safety/02_prediction/models/"<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the notebook will return the performance of the model on the hold-out data<br>

<img src="https://static.wixstatic.com/media/397bed_e0fd4340ff5f40de876b26f0fb7e1f83~mv2.png/v1/fill/w_610,h_610,al_c,q_85,usm_0.66_1.00_0.01/Grab%20EDM_Safety.webp"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />
