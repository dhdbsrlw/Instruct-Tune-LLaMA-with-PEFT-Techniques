# This code is implemented based on 'truthful_qa_scenario.py'
# 이 코드는 TruthfulQA benchmark dataset 을 학습시키기 위하여 데이터를 전처리하는 코드입니다.

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


# Dataset download

def download_dataset(output_path):  # output_path: str
    """Downloads the TruthfulQA dataset."""
    # Download the raw data
    data_dir = os.path.join(output_path, "data")
    url = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
    ensure_directory_exists(data_dir)  # output_path 에 해당하는 폴더가 존재하지 않을 경우 생성
    ensure_file_downloaded(source_url=url, target_path=os.path.join(
        data_dir, DATASET_FILE_NAME))  # 오류시 수정 필요

# Load dataset


def load_dataset(output_path: str) -> List[Dict[str, Any]]:
    """Loads the dataset downloaded in download_dataset()."""
    file_path = os.path.join(output_path, "data", DATASET_FILE_NAME)
    data = []
    with open(file_path, encoding="utf-8") as f:
        # Skip headers
        csv_reader = csv.reader(f)
        next(csv_reader)
        # Loop through the file
        for _type, category, question, best_answer, correct_answers, incorrect_answers, source in csv_reader:
            data_point = {
                "category": category,
                "question": question,
                "best_answer": best_answer,
                "correct_answers": correct_answers,
                "incorrect_answers": incorrect_answers,
                "source": source,
            }
            data.append(data_point)
    return data


# Get train/val instances

def get_instances(output_path: str):
    """Returns the instances for this scenario."""

    def format_str(unformatted_str: str) -> str:
        formatted_str = unformatted_str.strip()
        if formatted_str[-1] != ".":
            formatted_str = formatted_str + "."
        return formatted_str

    def split_multiple_answer_string(multiple_answers: str, seperator=";") -> List[str]:
        return [format_str(a.strip()) for a in multiple_answers.split(seperator) if a.strip()]

    def get_references(best_answer: str, incorrect_answers: List[str]):
        # Prepare the references list
        # 튜플로 구성된 리스트
        references = [f"INCORRECT : {ans}" for ans in incorrect_answers]
        # 리스트의 마지막에 Best Answer 추가
        references.append(f"CORRECT : {best_answer}")

        # To ensure that we have some variety at where the option with the correct answer
        # appears (A, B, C etc.) we use ascii value of the first character of the best_answer
        # string (ord) and use ord mod the list length to rotate the references list.
        # 한 줄 요약: 선지 간 순서 섞기
        k = ord(best_answer[0]) % len(references)
        references = references[k:] + references[:k]
        return references

    def get_split_instances(split, data):
        instances = []  # 데이터 리스트

        for dt in data:
            # Format the fields of the question
            question: str = dt["question"].strip()
            best_answer: str = format_str(dt["best_answer"])
            incorrect_answers: List[str] = split_multiple_answer_string(
                dt["incorrect_answers"])

            # Prepare the instance - 문제의 구간, reference 들로 하나의 instance 를 구성한다.
            references = get_references(best_answer, incorrect_answers)
            instance = [f"- Input: {question} \
        - Reference: {references} \
        - Split: {split}"]
            instances.append(instance)

        return instances

    # Execution

    download_dataset(OUTPUT_PATH)
    data = load_dataset(OUTPUT_PATH)
    # print(data[0])
    split_k = int(len(data) * TRAIN_RATIO)
    # print(split_k) # 현재: 전체 데이터셋의 80% 분리
    train_instances = get_split_instances(TRAIN_SPLIT, data[:split_k])
    valid_instances = get_split_instances(VALID_SPLIT, data[split_k:])
    # print(train_instances[0])

    return train_instances + valid_instances

#######################################################
