import glob
import chardet
import pandas as pd
import os
import logging
from utils import Utility

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Utility().create_folder('Logs')
params = Utility().read_params()

main_log_folderpath = params['logging_folder_paths']['main_log_foldername']
data_loading_filename = params['logging_folder_paths']['data_loading_filename']

file_handler = logging.FileHandler(os.path.join(main_log_folderpath, data_loading_filename))

formatter = logging.Formatter(
    '%(asctime)s : %(levelname)s : %(filename)s : %(message)s')

file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def combine_csvs_with_season():
    """ Combine all csv datasets into a single file with season tagging using two loops """
    # Get the path to the script's parent directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Move up two directories to reach the project root (since you are in src/data)
    root_dir = os.path.abspath(os.path.join(script_dir, '..'))

    csv_pattern = os.path.join(root_dir, 'Data', 'raw', "*.csv")
    filenames = glob.glob(csv_pattern)

    # Read and merge CSVs using original code
    df_master = pd.DataFrame()
    for filename in filenames:
        df_trip = pd.read_csv(
            filename,
            sep=';',
            encoding=chardet.detect(open(filename, 'rb').read())['encoding']
        )
        df_master = pd.concat([df_master, df_trip])

    # Construct the target path: project root's Data/processed/AllTrips.csv
    target_dir = os.path.join(root_dir, 'Data', 'processed')
    os.makedirs(target_dir, exist_ok=True)
    df_master.to_csv(os.path.join(target_dir, 'AllTrips.csv'), index=False)

    return df_master

if __name__ == "__main__":

    combine_csvs_with_season()
