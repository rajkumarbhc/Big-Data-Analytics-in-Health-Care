# Mortality Prediction in ICU Big Data Group Project

Note: Please download the zipped files which contains the code for this project. 

Data source: MIMIC-III.

## Introduction

The repository contains final CSE-6250 project code created by Team 03.  The entire project development cycles were divided into two stages. Feature Engineering, Model Training.

## Setup and Data Preprocessing and Model Design


(i)  Feature Engineering as part of preprocessing:

Environment Specification:

Hadoop HDFS, and PySpark on Local Docker MAC OS environment (1 Master node and 2 worker nodes, each with 900 GB space, 14 GB RAM, and 4 processors).

Using conda, create the environment defined in (environment.yml) 

Use following script to generate feature rich data for the model:

```
python -m feature_builder.prepare_data {mimic3-datasets} {output_folder}

Ex. 

python -m feature_builder.prepare_data mimic3-db/ my_generated_data/
```


We loaded following tables into HDFS:

	- ADMISSIONS.csv
	- CHARTEVENTS.csv
	- DIAGNOSES_ICD.csv
	- D_ICD_DIAGNOSES.csv
	- ICUSTAYS.csv
	- LABEVENTS.csv
	- OUTPUTEVENTS.csv
	- PATIENTS.csv

feature_builder.prepare_data script: will take care of the orchestration process for filtering a building the features.

On the other side repository script contains the data layer implementation done with pyspark. 


(ii) Model Architecture:

Use following script to run the model:

The scripts will be using the pre processed data sets located in the data directory.

From the project root folder run:
```
python models/Keras_ANN_mortality_prediction.py
python models/Keras_CNN_mortality_prediction.py

There are two Jupytor Notebook files:
RandomForest_Model_Mortality_Prediction_full_data.ipynb
LogisticRegression_Model_Mortality_Prediction_full_data.ipynb
```

Here are the steps followed to create and train the model:
- Step 1:
  Extracted the feature rich data generated part of Preprocessing step above using pandas numpy library.

    ```
    path_or_buf = "../data/"    
    np.savetxt(path_or_buf +'test_train_y.txt', train_y)  
    np.savetxt(path_or_buf +'test_val_y.txt', val_y)    
    np.savetxt(path_or_buf +'test_test_y.txt', test_y) 
    train_X.tofile(path_or_buf +'test_train_X.txt',sep="", format="%s")   
    val_X.tofile(path_or_buf +'test_val_X.txt',sep="", format="%s")    
    test_X.tofile(path_or_buf +'test_test_X.txt',sep="", format="%s")
    ```


- Step 2:
Applied sklearn 's SimpleImputer function to replace 'nan' (null) values, and apply mean strategy to prepare data for Machine Learning model applications.

- Step 3:
Train Machine Learning Model, and Validate Model. Run Model for different epochs and modify hyper parameters to tune Model Performance.  

In Phase 1 and Phase 2, we trained feature rich MIMIC III data using 4 models.
1. Logistic Regression.
2. Random Forest. 
3. Keras ANN/CNN deep Learning models. 

The final code used for training data set is available at below folder.

```
—  /models - Model code
- /models/notebooks - Jupyter Notebook code.  To view the details, the files need to be opened in Jupyter Notebook environment. 
—  /model_output - The best models saved as pickle file for each of the four models.    ANN_Best_model.sav, CNN_Best_model.sav, LR_Best_model.sav, RF_Best_model.sav.

—  /models/reports - the logs from our model training & parameter tuning.
-  /data_full_extracts  - contains feature rich data for training the models.  
train_X.txt,train_y.txt  {training set}
test_X.txt,test_y.txt {test set}
val_X.txt, val_y.txt {validate set}
```

 The Best models files saved for testing

```
—  /model_output - The best models saved as pickle file for each of the four models.    ANN_Best_model.sav, CNN_Best_model.sav, LR_Best_model.sav, RF_Best_model.sav.

```



We finalized two Keras deep learning  framework models in Python for our machine learning model evaluation in Phase 2.  Keras is an open source machine learning API framework that can run on top of Theano, Tensorflow, or CNTK framework. Karas APIs are simple to use and to understand the Machine Learning problems, yet we get the full power of its underlying powerful deep learning architecture.  


ANN/CNN Keras Models -  We tried ‘binary_crossentropy’ and ‘mean_squired_error’ as two loss functions, and ‘adam’, and sgd as two optimizers. We noticed that ‘sgd’ optimizer, and ‘mean_squired_error’ loss function yielded expected AUC/ROC, F1 score, and Accuracy, and we finalized our model based on this finding. We trained model  using train data size of 14681* 714, validated using 3222* 714, and tested using test set of size 3236* 714. The training dataset was of reasonable size to run on a local laptop MAC OS environment, therefore we chose to train model in Spyder/Anaconda Python 3.6 environment, and capture the results log in HTML form for further review. 

- Step 4:
Evaluated Model Performance using Confusion Matrix, AUC/ROC Curve, Loss Curve, F1/Accuracy values.  We adjusted lr(regularization parameter value 0.01, and 0.001, and tried loss function 'binary_crossentropy', and 'mean_squared_erros'.  We tried 'Adam','sgd' optimizers for ANN/CNN, but finalized the model using 'sgd',and 'mean_squared_error' specifics.
