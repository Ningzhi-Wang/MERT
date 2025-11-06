import io
import os

import boto3
import numpy as np
import soundfile as sf

def get_audio_file(file_path: str) -> bytes:
    if os.get_env("DATASET_LOCATION", "LOCAL") == "S3":
        wav, sr = read_from_s3(file_path)
    else:
        wav, sr = sf.read(file_path)
    return wav, sr

def read_from_s3(file_path: str) -> bytes:
    aws_key = os.getenv("AWS_ACCESS_KEY")
    aws_secret = os.getenv("AWS_SECRET_KEY")
    assert aws_key is not None, "AWS_ACCESS_KEY not set"
    assert aws_secret is not None, "AWS_SECRET_KEY not set"
    s3 = boto3.client(
        "s3",
        endpoint_url="https://ceph-private-object-rgw.comp-research.qmul.ac.uk",
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret,
    )

    file_buffer = io.BytesIO()
    s3.download_fileobj(
        "c4dm-01",
        file_path,
        file_buffer
    )
    file_buffer.seek(0)
    wav, sr = sf.read(file_buffer)
    return wav, sr