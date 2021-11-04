# Import libraries
import os
import argparse
import pandas as pd
from azureml.core import Run


# Get parameters
parser = argparse.ArgumentParser()
parser.add_argument("--input-data", type=str, dest='raw_dataset_id', help='raw dataset')
parser.add_argument('--prepped-data', type=str, dest='prepped_data', default='prepped_data', help='Folder for results')
args = parser.parse_args()
save_folder = args.prepped_data

# Get the experiment run context
run = Run.get_context()

# load the data (passed as an input dataset)
print("Loading Data...")
telecom = run.input_datasets['raw_data'].to_pandas_dataframe()

# Log raw row count
row_count = (len(telecom))
run.log('raw_rows', row_count)

# data cleaning
telecom = telecom.dropna()

telecom['voice_mail_plan'] = telecom['voice_mail_plan'].map(lambda x: x.strip())
telecom['intl_plan'] = telecom['intl_plan'].map(lambda x: x.strip())
telecom['churned'] = telecom['churned'].astype('str') 
telecom['churned'] = telecom['churned'].map(lambda x: x.strip())
telecom = telecom.replace(['True.', 'False.'], ['True','False']) 


# Log processed rows
row_count = (len(telecom))
run.log('processed_rows', row_count)

# Save the prepped data
print("Saving Data...")
os.makedirs(save_folder, exist_ok=True)
save_path = os.path.join(save_folder,'data.csv')
telecom.to_csv(save_path, index=False, header=True)

# End the run
run.complete()
