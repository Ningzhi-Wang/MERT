import torch
import pytest

from .utils import read_from_s3

################################################
# Unit tests for S3 Audio File Reading  
################################################

def test_access():
    wav, sr = read_from_s3("acw713/datasets/mert_audios/000200.wav")
    assert wav.ndim == 2, "Audio should have 2 dimensions (channels, samples)"
    assert sr == 24000, "Sample rate should be 24000 Hz"




