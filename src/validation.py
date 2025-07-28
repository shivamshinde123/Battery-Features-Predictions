import pandas as pd
from pydantic import ValidationError, confloat, conint, create_model
import os
from src.utils import Utility

logger = Utility().setup_logger()

class DataValidation:

    """This class helps validate the data types of features in input data"""

    def __init__(self):
        # Get the path to the script's parent directory
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        # Move up two directories to reach the project root (since you are in src)
        self.root_dir = os.path.abspath(os.path.join(self.script_dir, '..'))
        # load data
        self.df = pd.read_csv(os.path.join(self.root_dir, 'Data', 'processed', 'AllTrips.csv'), sep=';')

    def validate_data(self):

        """This method is used to validate the input datatypes"""

        try:
            # Map pandas types to Pydantic types
            dtype_mapping = {
                "float64": confloat(allow_inf_nan=True),
                "int64": conint(ge=-2 ** 31, le=2 ** 31 - 1),
            }

            # Build a dictionary for datatype and defaults
            field_definitions = {}
            for col, dtype in self.df.dtypes.items():
                dtype_str = str(dtype)
                ptype = dtype_mapping.get(dtype_str, str)
                field_definitions[col] = (ptype, ...)  # combine type + required flag

            EVDataModel = create_model("EVDataModel", **field_definitions)

            # Validate each row of your CSV
            invalid_rows = []
            for i, row in enumerate(self.df.to_dict(orient='records')):
                try:
                    EVDataModel(**row)  # this attempts validation
                except ValidationError as e:
                    invalid_rows.append((i, e.errors()))

            # Print results
            if not invalid_rows:
                logger.info('All rows have valid datatypes.')
                return True
            else:
                logger.info(f"Found {len(invalid_rows)} invalid rows:")
                for idx, errs in invalid_rows[:5]:
                    logger.log(f"Row {idx} errors: {errs}")
                return False

        except Exception as e:
            logger.error('Unexpected failure in validation', exc_info=e)
            raise


if __name__ == "__main__":


    dv = DataValidation()
    if dv.validate_data():
        logger.info('Data Validation Successful!')
    else:
        logger.warning("Data Validation Failed!!!")

