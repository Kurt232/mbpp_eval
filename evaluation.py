import ast
import pandas as pd
import numpy as np
import itertools

from typing import *
from tqdm.auto import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from .data import stream_jsonl
from .execution import check_correctness

import re

def read_dataset(
    data_file: str = None,
) -> Dict:
    """
    Reads a dataset and returns a dictionary of tasks.
    """
    dataset = {task["task_id"]: task for task in stream_jsonl(data_file)}

    return dataset


def estimate_pass_at_k(
        num_samples: Union[int, List[int], np.ndarray],
        num_correct: Union[List[int], np.ndarray],
        k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def evaluate_functional_correctness(
        df: pd.DataFrame = None,
        n_workers: int = 32,
        timeout: float = 10.0,
        problem_file: str = "./mbpp_test.jsonl",
        k: List[int] = [1, 10, 100],
        example_test: bool = False,
):
    """
    Evaluates the functional correctness of a model.
    """
    if example_test:
        print("Example test...")

    code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)
    
    def extract_last_code_block(text):
        matches = code_pattern.findall(text)
        return matches[-1] if matches else None
    
    df["code"] = df["generation"].apply(extract_last_code_block)
    if len(df[df["code"].notnull()]) < len(df):
        print(f" format missed: {len(df[df['code'].notnull()])} / {len(df)}")

    sample_jsonl = df[["generation", "code", "task_id"]]
    sample_jsonl = sample_jsonl.to_dict(orient="records")

    problems = read_dataset(problem_file)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:

        futures = []
        completion_id = Counter()
        n_samples = 0
        results = defaultdict(list)

        print("Reading samples...")
        for sample in tqdm(sample_jsonl):
            task_id = sample["task_id"]
            sample["task_id"] = task_id
            if sample["code"] is None:
                print(f"{task_id} is format error.")
                sample["code"] = ""
            sample["test_code"] = sample["code"] + "\n" + "\n".join(problems[task_id]["test"])

            if "completion_id" in sample:
                completion_id_ = sample["completion_id"]
            else:
                completion_id_ = completion_id[task_id]
            args = (task_id, sample, timeout, completion_id_)
            future = executor.submit(check_correctness, *args)
            futures.append(future)
            completion_id[task_id] += 1
            n_samples += 1

        if len(completion_id) == len(problems):
            evaluate_pass_at_k = True
        else:
            evaluate_pass_at_k = False

        print("Running test suites...")
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            results[result["task_id"]].append((result["completion_id"], result))

    # Calculate pass@k.
    total, correct = [], []
    for result in results.values():
        passed = [r[1]["passed"] for r in result]
        total.append(len(passed))
        correct.append(sum(passed))
    total = np.array(total)
    correct = np.array(correct)
    if evaluate_pass_at_k:
        ks = k
        pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
                     for k in ks if (total >= k).all()}
        print(pass_at_k)
    else:
        print("Total:", np.sum(total))
        print("Correct:", np.sum(correct))
    return pass_at_k
