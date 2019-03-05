import utils
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta as subtract_time


# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    
    '''
    TODO: This function needs to be completed.
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv')
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 a

    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''
    
    
    events_copy = events.copy()
    events_copy = events_copy.assign(Dead=events_copy['patient_id'].isin(mortality['patient_id']).astype(int))
    alive_df = events_copy.copy()
    alive_df = alive_df.loc[alive_df['Dead'] == 0]
    alive_df = alive_df.drop(['Dead', 'event_id', 'event_description', 'value'], axis=1)
    alive_df = alive_df.drop_duplicates()
    alive_df['timestamp'] =pd.to_datetime(alive_df.timestamp)
    alive_df2 = alive_df.loc[alive_df.groupby('patient_id').timestamp.idxmax()]
    
    dead_df = mortality.copy()
    dead_df= dead_df[['patient_id','timestamp']]
    dead_df['timestamp']= pd.to_datetime(dead_df['timestamp'])
    dead_df['timestamp']= dead_df['timestamp']-subtract_time(days = 30)
    
    indx_date = pd.concat([alive_df2,dead_df]).reset_index(drop=True)
    indx_date = indx_date.rename(columns={'timestamp': 'indx_date'})
    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)
    
    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 b

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''

    
    JoinDataFrame = pd.merge(events, indx_date, on = ['patient_id'])
    #print(JoinDataFrame)

    JoinDataFrame['indx_date'] = pd.to_datetime(JoinDataFrame['indx_date'])
    JoinDataFrame['timestamp'] = pd.to_datetime(JoinDataFrame['timestamp'])
    
    filtered_events = JoinDataFrame.copy()
    #filtered_events["diffrence"] = filtered_events['indx_date'] - filtered_events['timestamp']
    filtered_events = filtered_events[(filtered_events.timestamp <= filtered_events.indx_date) & (filtered_events.timestamp >= filtered_events.indx_date-subtract_time(days = 2000))]
    
    #filtered_events = filtered_events.drop(['event_description', 'timestamp', 'indx_date'], axis=1)
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)
    return filtered_events

    


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 c

    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and count to calculate feature value
    4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return filtered_events
    '''
    filtered_eventsdf = pd.merge(filtered_events_df, feature_map_df, on = 'event_id')
    #filtered_eventsdf = filtered_eventsdf.rename(columns={'event_id': 'idx'})
    filtered_eventsdf = filtered_eventsdf[['patient_id','idx','value']]
    filtered_eventsdf = filtered_eventsdf[pd.notnull(filtered_eventsdf['value'])]
    
    
    DiagDrugGroup = filtered_eventsdf[filtered_eventsdf['idx'] < 2680]
    #print(DiagDrugGroup)
    add_Diag_Drug = DiagDrugGroup.groupby(['patient_id','idx']).agg('sum')
    add_Diag_Drug.reset_index(inplace = True)

    add_Diag_DrugMax = add_Diag_Drug.groupby(['idx']).agg('max')
    add_Diag_DrugMax.reset_index(inplace = True)
    #print(add_Diag_DrugMax)
    add_Diag_DrugMax = add_Diag_DrugMax.rename(columns = {"value": "max"}) 
    add_Diag_DrugMax = add_Diag_DrugMax.drop(['patient_id'], axis=1)

    add_Diag_Drug = pd.merge(add_Diag_Drug, add_Diag_DrugMax, on = 'idx')
    add_Diag_Drug["value2"] = add_Diag_Drug["value"]/add_Diag_Drug["max"]
    add_Diag_Drug = add_Diag_Drug.drop(['value', 'max'], axis=1)
    add_Diag_Drug = add_Diag_Drug.rename(columns = {"value2": "feature_value", "idx": "feature_id"})


    LabGroup = filtered_eventsdf[filtered_eventsdf['idx'] >= 2680]
    
    LabCount = LabGroup.groupby(['patient_id','idx']).agg('count')
    LabCount.reset_index(inplace = True)
    
    LabCountMax = LabCount.groupby(['idx']).agg('max')
    LabCountMax.reset_index(inplace = True)

    LabCountMax = LabCountMax.rename(columns = {"value": "max"}) 
    LabCountMax = LabCountMax.drop(['patient_id'], axis=1)

    LabCount = pd.merge(LabCount, LabCountMax, on = 'idx')
    LabCount["value2"] = LabCount["value"]/LabCount["max"]
    LabCount = LabCount.drop(['value', 'max'], axis=1)
    LabCount = LabCount.rename(columns = {"value2": "feature_value", "idx": "feature_id"})


    aggregated_events = pd.concat([add_Diag_Drug, LabCount]).reset_index(drop = True)
    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)
    
    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)
    

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''
    patient_features = aggregated_events.groupby('patient_id')[['feature_id','feature_value']].apply(lambda x: [tuple(x) for x in x.values]).to_dict()
    
    
    events_copy = events.copy()
    events_copy = events_copy.assign(Dead=events_copy['patient_id'].isin(mortality['patient_id']).astype(int))
    alive_df = events_copy.copy()
    alive_df = alive_df.loc[alive_df['Dead'] == 0]
    dead_df = events_copy.copy()
    dead_df = dead_df.loc[dead_df['Dead'] == 1]
    
    new_df = pd.concat([alive_df, dead_df]).reset_index(drop = True)
    new_df = new_df.drop(['event_id', 'event_description', 'timestamp', 'value'], axis=1) 
    
    mortality = pd.Series(new_df.Dead.values, index = new_df.patient_id).to_dict()

    return patient_features, mortality

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
    '''
    TODO: This function needs to be completed

    Refer to instructions in Q3 d

    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')
    for patient_id in sorted(patient_features):
        feature_pairs_string = ""
        for feature in sorted(patient_features[patient_id]):
            feature_pairs_string += " " + str(int(feature[0])) + ":" + format(feature[1], '.6f')
        svm_light_str = str(mortality[patient_id]) + feature_pairs_string + " \n"
        deliverable1.write(bytes((svm_light_str),'UTF-8')); #Use 'UTF-8'
        deliverable2.write(bytes((str(int(patient_id)) + " " + svm_light_str),'UTF-8'));

def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()