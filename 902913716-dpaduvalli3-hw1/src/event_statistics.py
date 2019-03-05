import time
import pandas as pd
import numpy as np

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    '''
    TODO : This function needs to be completed.
    Read the events.csv and mortality_events.csv files. 
    Variables returned from this function are passed as input to the metric functions.
    '''
    events = pd.read_csv(filepath + 'events.csv')
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    return events, mortality

def event_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the event count metrics.
    Event count is defined as the number of events recorded for a given patient.
    '''
    events_copy = events.copy()
    events_copy = events_copy.assign(Dead=events_copy['patient_id'].isin(mortality['patient_id']).astype(int))
    alive_df = events_copy.copy()
    alive_df = alive_df.loc[alive_df['Dead'] == 0]
    dead_df = events_copy.copy()
    dead_df = dead_df.loc[dead_df['Dead'] == 1]

    deadEventidCount = dead_df['event_id'].count()
    deadUniquePatientId = dead_df['patient_id'].nunique()
    dead_df['event_freq'] = dead_df.groupby('patient_id')['patient_id'].transform('count')
    
    avg_dead_event_count = deadEventidCount/deadUniquePatientId
    max_dead_event_count = dead_df['event_freq'].max()
    min_dead_event_count = dead_df['event_freq'].min()

    aliveEventidCount = alive_df['event_id'].count()
    aliveUniquePatientId = alive_df['patient_id'].nunique()
    alive_df['event_freq'] = alive_df.groupby('patient_id')['patient_id'].transform('count')

    avg_alive_event_count = aliveEventidCount/aliveUniquePatientId
    max_alive_event_count = alive_df['event_freq'].max()
    min_alive_event_count = alive_df['event_freq'].min()

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the encounter count metrics.
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
    '''
    events_copy = events.copy()
    events_copy = events_copy.assign(Dead=events_copy['patient_id'].isin(mortality['patient_id']).astype(int))
    alive_df = events_copy.copy()
    alive_df = alive_df.loc[alive_df['Dead'] == 0]
    alive_df = alive_df.drop(['Dead', 'event_id', 'event_description', 'value'], axis=1)
    alive_df = alive_df.drop_duplicates()
    dead_df = events_copy.copy()
    dead_df = dead_df.loc[dead_df['Dead'] == 1]
    dead_df = dead_df.drop(['Dead', 'event_id', 'event_description', 'value'], axis=1)
    dead_df = dead_df.drop_duplicates()
    
    dead_timestamp = dead_df['timestamp'].count()
    dead_patientidCount = dead_df['patient_id'].nunique()
    dead_df['freq'] = dead_df.groupby('patient_id')['patient_id'].transform('count')
    
    avg_dead_encounter_count = dead_timestamp/dead_patientidCount
    max_dead_encounter_count = dead_df['freq'].max()
    min_dead_encounter_count = dead_df['freq'].min()

    alive_timestamp = alive_df['timestamp'].count()
    alive_patientidCount = alive_df['patient_id'].nunique()
    alive_df['freq'] = alive_df.groupby('patient_id')['patient_id'].transform('count')
    
    avg_alive_encounter_count = alive_timestamp/alive_patientidCount
    max_alive_encounter_count = alive_df['freq'].max()
    min_alive_encounter_count = alive_df['freq'].min()

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    TODO: Implement this function to return the record length metrics.
    Record length is the duration between the first event and the last event for a given patient. 
    '''
    events_copy = events.copy()
    events_copy['timestamp'] = pd.to_datetime(events_copy['timestamp'])
    recordDataFrame = events_copy[['patient_id','timestamp']].groupby('patient_id').agg([max,min])
    recordDataFrame['duration'] = recordDataFrame['timestamp']['max']-recordDataFrame['timestamp']['min']
    recordDataFrame = recordDataFrame['duration']
    death_df = recordDataFrame[mortality['patient_id']]
    death_df = death_df.dt.days  
    avg_dead_rec_len = death_df.mean()
    max_dead_rec_len = death_df.max()
    min_dead_rec_len = death_df.min()
    alive_df = recordDataFrame.drop(mortality['patient_id'])
    alive_df = alive_df.dt.days
    avg_alive_rec_len = alive_df.mean()
    max_alive_rec_len = alive_df.max()
    min_alive_rec_len = alive_df.min()
    
    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    '''
    DO NOT MODIFY THIS FUNCTION.
    '''
    # You may change the following path variable in coding but switch it back when submission.
    train_path = '../data/train/'

    # DO NOT CHANGE ANYTHING BELOW THIS ----------------------------
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute event count metrics: " + str(end_time - start_time) + "s"))
    print(event_count)

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute encounter count metrics: " + str(end_time - start_time) + "s"))
    print(encounter_count)

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute record length metrics: " + str(end_time - start_time) + "s"))
    print(record_length)
    
if __name__ == "__main__":
    main()
