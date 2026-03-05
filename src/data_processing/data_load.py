import os

import pandas as pd
import s3fs
from dotenv import load_dotenv


def data_laoding(set: str = "train") -> pd.DataFrame:
    # load the environment variables from the .env file
    load_dotenv(override=True)

    # load the dataset stored in the s3 bucket of MinIO
    fs = s3fs.S3FileSystem(
        key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
        token=os.getenv("AWS_SESSION_TOKEN"),
        client_kwargs={"endpoint_url": "https://" + os.getenv("AWS_S3_ENDPOINT")},
    )

    # load the train dataset
    set_path = os.getenv("AWS_BUCKET_NAME") + f"/{set}.csv"
    with fs.open(set_path, mode="rb") as f:
        df = pd.read_csv(f)

    return df
