import yaml
import os
import logging
import inspect

class Utility:

    def __init__(self, params_path=os.path.join("..", "params.yaml")) -> None:
        self.params_path = params_path

    def create_folder(self, folder_name):
        """This method is used to create folders that are required.

        Parameters
        -----------

        folder_name: Name of the folder that is needed to be created

        Returns
        --------
        None
        """

        try:
            # Creating a directory if it does not exist already
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

        except Exception as e:
            raise e

    def read_params(self):
        """This method is used to read the parameters yaml file and returns the loaded file object.

        Parameters
        -----------

        config_path: Path to the parameters yaml file

        Returns
        --------
        Loaded yaml file object: Returns the yaml file object.
        """

        try:
            # Reading params yaml file
            with open(self.params_path, 'r') as params_file:
                params = yaml.safe_load(params_file)

        except Exception as e:
            raise e

        else:
            return params

    def get_caller_filename(self):
        frame = inspect.stack()[2]
        full_path = frame.filename
        return os.path.splitext(os.path.basename(full_path))[0]

    def setup_logger(self):

        log_dir = os.path.join("..", "Logs")
        Utility().create_folder(log_dir)

        name = self.get_caller_filename() # file name with .py
        filename = os.path.join(log_dir, f"{name}.log")

        logger = logging.getLogger(name)
        if not logger.hasHandlers():
            logger.setLevel(logging.INFO)
            handler = logging.FileHandler(filename, mode='a')
            formatter = logging.Formatter(
                "%(asctime)s : %(levelname)s : %(filename)s : %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger
