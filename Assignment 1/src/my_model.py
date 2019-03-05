import utils
from sklearn.metrics import *
from sklearn.ensemble import AdaBoostRegressor
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from numpy import mean, array


#Note: You can reuse code that you wrote in etl.py and models.py and cross.py over here. It might help.
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

RANDOM_STATE = 545510477
'''
You may generate your own features over here.
Note that for the test data, all events are already filtered such that they fall in the observation window of their respective patients. Thus, if you were to generate features similar to those you constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.
IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
patient_id followed by a space and the corresponding feature in sparse format.
Eg of a line:
60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

Save the file as "test_features.txt" and save it inside the folder deliverables

input:
output: X_train,Y_train,X_test
'''
#def my_features():
#	#TODO: complete this
#	return None,None,None

'''
You can use any model you wish.

input: X_train, Y_train, X_test
output: Y_pred
'''
def my_classifier_predictions(X_train,Y_train):
#    model1 = RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=RANDOM_STATE).fit(X_train,Y_train)
#    Y_pred = model1.predict(X_train)
#    return Y_pred
    model1 = DecisionTreeRegressor(max_depth = 5)
    model2 = AdaBoostRegressor(model1, n_estimators= 400).fit(X_train,Y_train)
    Y_pred = model2.predict(X_train)
    return Y_pred

def get_acc_auc_kfold(X,Y,k=5): 
    kfoldCross_validation = KFold(n_splits = k, random_state = RANDOM_STATE)
    clf_lr_kfold = DecisionTreeRegressor()
    accracy_list =[]
    kfoldauc_list =[]
    for indecies_train,indecies_test in kfoldCross_validation.split(X):
        k = clf_lr_kfold.fit(X[indecies_train],Y[indecies_train])
        accracy = accuracy_score(k.predict(X[indecies_test]),Y[indecies_test])
        accracy_list.append(accracy)      
        accracy2 = roc_auc_score(k.predict(X[indecies_test]),Y[indecies_test])
        kfoldauc_list.append(accracy2)    
    kfold_accuracy = mean(accracy)
    kfold_accuracy_auc = mean(kfoldauc_list)
    print(kfold_accuracy)
    return kfold_accuracy,kfold_accuracy_auc


def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.5):
    kfoldCross_validation = ShuffleSplit(n_splits = iterNo, random_state = RANDOM_STATE, test_size = test_percent)
    clf_lr_kfold = DecisionTreeRegressor()
    accracy_list =[]
    kfoldauc_list =[]
    for indecies_train,indecies_test in kfoldCross_validation.split(X):            
        k = clf_lr_kfold.fit(X[indecies_train],Y[indecies_train])
        accracy = accuracy_score(k.predict(X[indecies_test]),Y[indecies_test])
        accracy_list.append(accracy) 
        accracy2 = roc_auc_score(k.predict(X[indecies_test]),Y[indecies_test])
        kfoldauc_list.append(accracy2)
    kfold_accuracy = array(accracy).mean()
    kfold_accuracy_auc = array(kfoldauc_list).mean()
    return kfold_accuracy,kfold_accuracy_auc


def main():
    X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
    Y_pred = my_classifier_predictions(X_train,Y_train)
    utils.generate_submission("../deliverables/features.train",Y_pred)
	#The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.
    X,Y = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
    print("Classifier: Decision Tree Regressor__________")
    acc_k,auc_k = get_acc_auc_kfold(X,Y)
    print(("Average Accuracy in KFold CV: "+str(acc_k)))
    print(("Average AUC in KFold CV: "+str(auc_k)))
    acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
    print(("Average Accuracy in Randomised CV: "+str(acc_r)))
    print(("Average AUC in Randomised CV: "+str(auc_r)))


if __name__ == "__main__":
    main()

	