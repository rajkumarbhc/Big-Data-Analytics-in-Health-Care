import os
import numpy as np
import pickle
import pandas as pd


PATH_TRAIN = "../data/mortality/train/"
PATH_VALIDATION = "../data/mortality/validation/"
PATH_TEST = "../data/mortality/test/"
PATH_OUTPUT = "../data/mortality/processed/"

def convert_icd9(icd9_object):
    code_string = str(icd9_object)
    x = 3
    if code_string[0].isalpha() and not code_string.startswith('V'):
        x = 4
    if x >= len(code_string):
        return code_string
    return code_string[0: x]

def build_codemap(df_icd9, transform):
    lengthofICD9Code = df_icd9['ICD9_CODE'].dropna()
    lengthofICD9Code = lengthofICD9Code.apply(transform)
    lengthofICD9Code = lengthofICD9Code.unique()
    return dict(zip(lengthofICD9Code, np.arange(len(lengthofICD9Code))))

def create_dataset(path, codemap, transform):
    mortalityDataFrame = pd.read_csv(os.path.join(path, "MORTALITY.csv"))
    diagnosesDataFrame = pd.read_csv(os.path.join(path, "DIAGNOSES_ICD.csv"))
    admissionDataFrame = pd.read_csv(os.path.join(path, "ADMISSIONS.csv"))
    diagnosesDataFrame['ICD9_CODE'] = diagnosesDataFrame['ICD9_CODE'].transform(transform)
    groupDiagbyVisit = diagnosesDataFrame.groupby(['HADM_ID'])
    groupVisitsbyPatient = admissionDataFrame.groupby(['SUBJECT_ID'])
    seq_data = []
    patient_ids = []
    labels = []
    for x, y in groupVisitsbyPatient:
        patient_ids.append(x)
        label = mortalityDataFrame.loc[mortalityDataFrame['SUBJECT_ID'] == x]['MORTALITY'].values[0]
        labels.append(label)
        stored = []
        y = y.sort_values(by=['ADMITTIME'])
        for i, visit in y.iterrows():
            diags = groupDiagbyVisit.get_group(visit['HADM_ID'])['ICD9_CODE'].values
            diags = list(filter(lambda a: a in codemap.keys(), diags))
            stored.append(list(map(lambda a: codemap[a], diags)))
        seq_data.append(stored)
    return patient_ids, labels, seq_data

def main():
	# Build a code map from the train set
	print("Build feature id map")
	df_icd9 = pd.read_csv(os.path.join(PATH_TRAIN, "DIAGNOSES_ICD.csv"), usecols=["ICD9_CODE"])
	codemap = build_codemap(df_icd9, convert_icd9)
	os.makedirs(PATH_OUTPUT, exist_ok=True)
	pickle.dump(codemap, open(os.path.join(PATH_OUTPUT, "mortality.codemap.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Train set
	print("Construct train set")
	train_ids, train_labels, train_seqs = create_dataset(PATH_TRAIN, codemap, convert_icd9)

	pickle.dump(train_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.train"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.train"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Validation set
	print("Construct validation set")
	validation_ids, validation_labels, validation_seqs = create_dataset(PATH_VALIDATION, codemap, convert_icd9)

	pickle.dump(validation_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(validation_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(validation_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.validation"), 'wb'), pickle.HIGHEST_PROTOCOL)

	# Test set
	print("Construct test set")
	test_ids, test_labels, test_seqs = create_dataset(PATH_TEST, codemap, convert_icd9)

	pickle.dump(test_ids, open(os.path.join(PATH_OUTPUT, "mortality.ids.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_labels, open(os.path.join(PATH_OUTPUT, "mortality.labels.test"), 'wb'), pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_seqs, open(os.path.join(PATH_OUTPUT, "mortality.seqs.test"), 'wb'), pickle.HIGHEST_PROTOCOL)

	print("Complete!")


if __name__ == '__main__':
	main()
