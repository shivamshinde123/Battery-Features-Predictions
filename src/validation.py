import pandas as pd
from pydantic import ValidationError, confloat, conint, create_model
import os
from utils import Utility

logger = Utility().setup_logger()

if __name__ == "__main__":

    # Load data
    # Get the path to the script's parent directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Move up two directories to reach the project root (since you are in src)
    root_dir = os.path.abspath(os.path.join(script_dir, '..'))
    data_path = os.path.join(root_dir, 'Data', 'processed', 'AllTrips.csv')

    df = pd.read_csv(data_path, sep=';')

    # Map pandas types to Pydantic types
    dtype_mapping = {
        "float64": confloat(allow_inf_nan=True),
        "int64": conint(ge=-2**31, le=2**31 -1),
    }

    # Build a dictionary for datatype and defaults
    field_definitions = {}
    for col, dtype in df.dtypes.items():
        dtype_str = str(dtype)
        ptype = dtype_mapping.get(dtype_str, str)
        field_definitions[col] = (ptype, ...)  # combine type + required flag

    EVDataModel = create_model("EVDataModel", **field_definitions)

    # Validate each row of your CSV
    invalid_rows = []
    for i, row in enumerate(df.to_dict(orient='records')):
        try:
            EVDataModel(**row)  # this attempts validation
        except ValidationError as e:
            invalid_rows.append((i, e.errors()))

    # Print results
    if not invalid_rows:
        logger.info('All rows have valid datatypes.')
    else:
        logger.info(f"Found {len(invalid_rows)} invalid rows:")
        for idx, errs in invalid_rows[:5]:
            logger.log(f"Row {idx} errors: {errs}")
