from sys import exc_info

import yaml
import os
import logging
import inspect

class Utility:

    def __init__(self) -> None:
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_dir = os.path.abspath(os.path.join(self.script_dir, '..'))
        self.params_path = os.path.join(self.root_dir, "params.yaml")

    def get_caller_filename(self):

        """This method is used to find out the name of the file in which the logger is currently running."""

        try:
            frame = inspect.stack()[2]
            full_path = frame.filename
            return os.path.splitext(os.path.basename(full_path))[0]
        except IndexError:
            logging.getLogger(__name__).exception("Cannot determine caller filename")
            raise

    def setup_logger(self):

        """This method is used to initialize the logger"""

        name = self.get_caller_filename()  # file name with .py
        logger = logging.getLogger(name)

        try:
            log_dir = os.path.join("..", "Logs")
            Utility().create_folder(log_dir)

            filename = os.path.join(log_dir, f"{name}.log")

            if not logger.hasHandlers():
                logger.setLevel(logging.INFO)
                handler = logging.FileHandler(filename, mode='a')
                formatter = logging.Formatter(
                    "%(asctime)s : %(levelname)s : %(filename)s : %(message)s"
                )
                handler.setFormatter(formatter)
                logger.addHandler(handler)

            return logger

        except OSError:
            logging.getLogger(__name__).exception(f"Failed setting up logger for {name}")
            raise


    def create_folder(self, folder_name):

        """
        This method is used to create folders that are required.
        """

        try:
            # Creating a directory if it does not exist already
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

        except OSError as e:
            logging.getLogger(__name__).exception(f"Failed to create folder {folder_name}")
            raise

    def read_params(self):

        """
        This method is used to read the parameters yaml file and returns the loaded file object.
        """

        try:
            # Reading params yaml file
            with open(self.params_path, 'r') as params_file:
                params = yaml.safe_load(params_file)

        except FileNotFoundError:
            logging.getLogger(__name__).exception(f"params.yaml not found at {self.params_path}")
            raise
        except yaml.YAMLError:
            logging.getLogger(__name__).exception(f"Invalid YAML in {self.params_path}")
            raise

        else:
            return params

