import models_partc
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.linear_model import LogisticRegression
from numpy import mean, array
from sklearn.metrics import *

import utils

# USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

# USE THIS RANDOM STATE FOR ALL OF YOUR CROSS VALIDATION TESTS, OR THE TESTS WILL NEVER PASS
RANDOM_STATE = 545510477

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y,k=5):
	#TODO:First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the folds
    
    kfoldCross_validation = KFold(n_splits = k, random_state = RANDOM_STATE)
    clf_lr_kfold = LogisticRegression()
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


#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
	#TODO: First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the iterations
    
    kfoldCross_validation = ShuffleSplit(n_splits = iterNo, random_state = RANDOM_STATE, test_size = test_percent)
    clf_lr_kfold = LogisticRegression()
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
	X,Y = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	print("Classifier: Logistic Regression__________")
	acc_k,auc_k = get_acc_auc_kfold(X,Y)
	print(("Average Accuracy in KFold CV: "+str(acc_k)))
	print(("Average AUC in KFold CV: "+str(auc_k)))
	acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
	print(("Average Accuracy in Randomised CV: "+str(acc_r)))
	print(("Average AUC in Randomised CV: "+str(auc_r)))

if __name__ == "__main__":
	main()

