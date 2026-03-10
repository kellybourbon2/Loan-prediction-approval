import os

import pandas as pd
import s3fs
from dotenv import load_dotenv


class DataLoader:
    def __init__(self):
        """Initialize S3 connection once"""
        load_dotenv(override=True)

        self.fs = s3fs.S3FileSystem(
            key=os.getenv("AWS_ACCESS_KEY_ID"),
            secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
            token=os.getenv("AWS_SESSION_TOKEN"),
            client_kwargs={"endpoint_url": "https://" + os.getenv("AWS_S3_ENDPOINT")},
        )
        self.bucket = os.getenv("AWS_BUCKET_NAME")

    def load(self, set: str = "train") -> pd.DataFrame:
        """Load any dataset by name from S3"""
        path = f"{self.bucket}/{set}.csv"
        with self.fs.open(path, mode="rb") as f:
            return pd.read_csv(f)

    def load_train(self) -> pd.DataFrame:
        """loading of training set"""
        return self.load("train")

    def load_test(self) -> pd.DataFrame:
        """loading of testing set (useless for us)"""
        return self.load(
            "test"
        )  # loading of test set (useless bc test has no target, rather we split training)


def data_loading(set: str = "train") -> pd.DataFrame:
    """Wrapper function to load a dataset from S3"""
    loader = DataLoader()
    return loader.load(set)
