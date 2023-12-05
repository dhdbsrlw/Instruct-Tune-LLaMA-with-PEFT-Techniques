# This code is implemented based on 'narrative_qa_scenario.py'
# 이 코드는 NarrativeQA benchmark dataset 을 학습시키기 위하여 데이터를 전처리하는 코드입니다.

import os
import random
import csv
from typing import List, Dict

from .common import shell, ensure_directory_exists, ensure_file_downloaded

""" Data splits """
TRAIN_SPLIT: str = "train"
VALID_SPLIT: str = "valid"
TEST_SPLIT: str = "test"
EVAL_SPLITS: List[str] = [VALID_SPLIT, TEST_SPLIT]
ALL_SPLITS: List[str] = [TRAIN_SPLIT] + EVAL_SPLITS


def get_context(summary: str, question: str):
    """
    We follow the format from https://arxiv.org/abs/2005.14165.
    For more details, see the examples in Appendix G.
    """
    if question[-1] != "?":
        question = question + "?"

    # PassageQuestionInput format in 'scenario.py'
    passage = summary
    # question = question
    passage_prefix: str = "Summary: "
    question_prefix: str = "Question: "
    separator: str = "\n"

    context = passage_prefix + passage + separator + question_prefix + question
    return context


def get_split_instances(summaries_file, qaps_file, split):  # str, str, str
    """
    Helper for generating instances for a split.
    Args:
      summaries_file (str): File path for summaries (summaries.csv)
      qaps_file (str): File path for the question answer pairs (qaps.csv)
      split (str): Split (one of "train", "valid" or "test")

    Returns:
      List[Instance]: Instances for the specified split
    """

    split_instances = []  # 기존 Instance List
    split_summaries: Dict[str, Dict[str, str]] = {}

    with open(summaries_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["set"] != split:
                continue
            split_summaries[row["document_id"]] = row

    doc_id_to_question_rows: Dict[str, List[Dict[str, str]]] = {}
    with open(qaps_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["set"] != split:
                continue
            document_id: str = row["document_id"]
            if document_id in doc_id_to_question_rows:
                doc_id_to_question_rows[document_id].append(row)
            else:
                doc_id_to_question_rows[document_id] = [row]

    for document_id, rows in doc_id_to_question_rows.items():
        row = random.choice(rows)
        summary: str = split_summaries[document_id]["summary"]
        question: str = row["question"]
        answer1: str = row["answer1"]
        answer2: str = row["answer2"]

        # 수정 필요
        input = get_context(summary.strip(), question.strip())
        # references = []
        # references.append(f"CORRECT : {answer1}")
        # references.append(f"CORRECT : {answer2}")
        # split = split

        instance = [
            f"{input} \nAnswer: \n(Target Completion)\n {answer1}\n or\n {answer2}"]
        # instance = [f"- Input: {input} \n - Target Completion: {references} \n - Split: {split}"]
        # 전체 instance 목록에 추가
        split_instances.append(instance)

    return split_instances


def get_instances(output_path):
    data_path = os.path.join(output_path, "data")
    ensure_directory_exists(data_path)

    repo_url: str = "https://github.com/deepmind/narrativeqa/archive/master.zip"
    repo_path: str = os.path.join(data_path, "narrativeqa-master")

    ensure_file_downloaded(source_url=repo_url,
                           target_path=repo_path, unpack=True)

    # We will use the summaries, and the corresponding question and answer pairs.
    summaries_file: str = os.path.join(
        repo_path, "third_party", "wikipedia", "summaries.csv")
    qaps_file: str = os.path.join(repo_path, "qaps.csv")

    random.seed(0)  # we randomly pick one question per document
    instances = []
    for split in ALL_SPLITS:
        # get_split_instances 위에서 정의
        instances.extend(get_split_instances(
            summaries_file=summaries_file, qaps_file=qaps_file, split=split))

    return instances


############################################################################

# 실제 코드 실행

output_path = "/content/drive/MyDrive/helm"
result = get_instances(output_path)
print(result[0])
