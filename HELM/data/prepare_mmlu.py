"""Implementation derived from prepare_dolly.py"""

import sys
from pathlib import Path

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import pandas as pd
import numpy as np

import torch
from torch.utils.data import random_split
from lit_llama.tokenizer import Tokenizer
from tqdm import tqdm

from datasets import load_dataset

DATA_FILE = "cais/mmlu" 
DATA_FILE_NAME = "mmlu_data_cleaned.json"
IGNORE_INDEX = -1


def prepare(
    destination_path: Path = Path("data/mmlu"), 
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    max_seq_length: int = 1024,
    seed: int = 42,
    mask_inputs: bool = False,  # as in alpaca-lora
) -> None:
    """Prepare the Dolly dataset for instruction tuning.
    
    The output is a training and validation dataset saved as `train.pt` and `val.pt`,
    which stores the preprocessed and tokenized prompts and labels.
    """

    # 새로운 폴더 생성
    destination_path.mkdir(parents=True, exist_ok=True)
    # 데이터 다운로드
    mmlu = donwload()
    
    # 토크나이저 로드
    tokenizer = Tokenizer(tokenizer_path)
    
    # TRAIN/ VAL/ TEST 데이터 분리
    train_set = pd.DataFrame(mmlu["auxiliary_train"])
    val_set = pd.DataFrame(mmlu["validation"])
    test_set = pd.DataFrame(mmlu["test"])
    
    # Answer 열의 숫자를 알파벳으로 변환
    answer_list = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E'}

    train_set['answer'] = train_set['answer'].map(answer_list)
    val_set['answer'] = val_set['answer'].map(answer_list)
    test_set['answer'] = test_set['answer'].map(answer_list)
    
    # Choices 열의 선지를 정해진 형식에 맞게 고치기
    alphabet = ['A', 'B', 'C', 'D'] # 답은 항상 4개(A, B, C, D) 중에 하나

    ## train
    choices_column = train_set['choices']

    for i, choice_list in enumerate(choices_column):
        labeled_choices = [f"{alphabet[j]}. {choice}" for j, choice in enumerate(choice_list)]
        labeled_choices_2 = "\n".join(labeled_choices)
        train_set.at[i, 'choices'] = labeled_choices_2  

    ## val
    choices_column = val_set['choices']

    for i, choice_list in enumerate(choices_column):
        labeled_choices = [f"{alphabet[j]}. {choice}" for j, choice in enumerate(choice_list)]
        labeled_choices_2 = "\n".join(labeled_choices)
        val_set.at[i, 'choices'] = labeled_choices_2  


    ## test
    choices_column = test_set['choices']

    for i, choice_list in enumerate(choices_column):
        labeled_choices = [f"{alphabet[j]}. {choice}" for j, choice in enumerate(choice_list)]
        labeled_choices_2 = "\n".join(labeled_choices)
        test_set.at[i, 'choices'] = labeled_choices_2  

    # 각 데이터셋 전처리(Instruction Template 에 맞게 정리) 및 토크나이징
    print(f"train has {len(train_set):,} samples")
    print(f"val has {len(test_set):,} samples")

    # 각 행별 딕셔너리 화
    train_sample_list = []
    for index, row in train_set.iterrows():
        data_dict = {
        'question': row['question'],
        'answer': row['answer'],
        'choices': row['choices']
    }
        train_sample_list.append(data_dict)
        
    test_sample_list = []
    for index, row in test_set.iterrows():
        data_dict = {
        'question': row['question'],
        'answer': row['answer'],
        'choices': row['choices']
    }
        test_sample_list.append(data_dict)

    # 토크나이징        
    print("Processing train split ...")
    train_final_set = []
    for sample in tqdm(train_sample_list):
        tokenized_dict = prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) # 딕셔너리 반환
        train_final_set.append(tokenized_dict)
    torch.save(train_final_set, destination_path / "train.pt")

    # train_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(train_sample_list)]
    # torch.save(train_set, destination_path / "train.pt")

    print("Processing test split ...")
    test_final_set = []
    for sample in tqdm(test_sample_list):
        tokenized_dict = prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) # 딕셔너리 반환
        test_final_set.append(tokenized_dict)
    torch.save(test_final_set, destination_path / "test.pt")

    # test_set = [prepare_sample(sample, tokenizer, max_seq_length, mask_inputs) for sample in tqdm(test_sample_list)]
    # torch.save(test_set, destination_path / "test.pt")


def donwload():
    data = load_dataset(DATA_FILE, 'all') # 모든 카테고리의 데이터셋 
    print(data) # 데이터셋 구조 확인
     
    return data
   
   
def prepare_sample(example: dict, tokenizer: Tokenizer, max_length: int, mask_inputs: bool = True):
    full_prompt = generate_prompt(example)
    full_prompt_and_response = full_prompt + example["answer"]
    encoded_full_prompt = tokenize(tokenizer, full_prompt, max_length=max_length, eos=False)
    encoded_full_prompt_and_response = tokenize(tokenizer, full_prompt_and_response, eos=True, max_length=max_length)

    # The labels are the full prompt with response, but with the prompt masked out
    labels = encoded_full_prompt_and_response.clone()
    if mask_inputs:
        labels[:len(encoded_full_prompt)] = IGNORE_INDEX

    return {**example, "input_ids": encoded_full_prompt_and_response, "input_ids_no_response": encoded_full_prompt, "labels": labels}


def tokenize(tokenizer: Tokenizer, string: str, max_length: int, eos=True) -> torch.Tensor:
    return tokenizer.encode(string, bos=True, eos=eos, max_length=max_length)


def generate_prompt(example):
    # Generates a standardized message to prompt the model with an instruction, optional input and a 'response' field.

    return (
        f"Below is an instruction that describes a task, paired with an input that provides further context. "
        "Answer the question by replying A, B, C or D.\n\n"
        f"### Instruction:\n{example['question']}\n\n### Input:\n{example['choices']}\n\n### Answer:"
)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)