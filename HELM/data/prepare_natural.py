# This code is implemented based on 'natural_qa_scenario.py'
# 이 코드는 NaturalQA benchmark dataset 을 학습시키기 위하여 데이터를 전처리하는 코드입니다.

import csv
import os
from typing import Any, Callable, Dict, List, Optional, TypeVar

from .common import shell, ensure_directory_exists, ensure_file_downloaded


######################################################

# Dataset Config

""" Data storage"""
DATASET_FILE_NAME = "TruthfulQA.csv"  # 저장되는 데이터셋 파일명
OUTPUT_PATH = "/content/drive/MyDrive/helm"  # 추후 수정

""" Data splits """
TRAIN_SPLIT: str = "train"
VALID_SPLIT: str = "valid"
TEST_SPLIT: str = "test"
EVAL_SPLITS: List[str] = [VALID_SPLIT, TEST_SPLIT]
ALL_SPLITS: List[str] = [TRAIN_SPLIT] + EVAL_SPLITS

""" Number of examples """
# We mainly care about having enough test examples to ensure statistical significance;
# the remaining N-1000 instances become training examples.
DEFAULT_TEST_SIZE: int = 1000
TRAIN_RATIO: float = 0.8


#######################################################
