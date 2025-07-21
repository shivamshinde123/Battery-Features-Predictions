import glob
import chardet
import pandas as pd
import os
from utils import Utility

logger = Utility().setup_logger()

class Loader:

    def __init__(self):
        pass

    def combine_csvs_with_season(self):

        """ Combine all csv datasets into a single file with season tagging using two loops """

        try:
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

            logger.info("All the raw csv files were converted into one csv file.")

            return df_master

        except Exception as e:
            logger.error(f"Error occurred while loading the data: {e}")
            raise e


if __name__ == "__main__":

    loader = Loader()

    loader.combine_csvs_with_season()
