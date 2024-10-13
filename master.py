import json
import itertools
from tasks import compute_task
from chunked_pooling.chunked_eval_tasks import *

SKIP_EVAL_TASKS = [
    "ClimateFEVERChunked",
    "DBPediaChunked",
    "FEVERChunked",
    "HotpotQAChunked",
    "MSMARCOChunked",
]
OUTPUT_DIR = "results"
OVERWRITE = False


def load_existing_results():
    evaluated_key = set()
    # 读取已有的结果
    with open(f"{OUTPUT_DIR}/benchmark.json", "r") as f:
        try:
            content = json.load(f)
        except:
            print(
                "Invalid benchmark.json file, please check it. Perform full evaluation."
            )
            content = []
    for eval_res in content:
        eval_setting = {
            "task_name": eval_res["task_name"],
            "model_name": eval_res["model_name"],
            "chunking_strategy": eval_res["chunking_strategy"],
            "chunk_size": eval_res["chunk_size"],
        }
        key = json.dumps(eval_setting, sort_keys=True)
        assert (
            key not in evaluated_key
        ), f"{key} already exists, please check the benchmark.json file."
        evaluated_key.add(key)

    return evaluated_key, content


# 定义不合法组合的规则
def get_valid_setting_str(eval_setting, exist_results, overwrite=False):
    task_cls = eval_setting["task"]
    model_name = eval_setting["model_name"]
    strategy = eval_setting["strategy"]
    chunk_size = eval_setting["chunk_size"]

    # 规则1: semantic candhunking策略下chunk size无影响
    if "semantic" in strategy:
        eval_setting["chunk_size"] = -1

    # 规则2: bce-embedding-base_v1最长上下文为512, 因此无法使用late_chunking
    if "bce-embedding-base_v1" in model_name and strategy == "late_chunking":
        return None

    setting_key = json.dumps(eval_setting, sort_keys=True)
    # 规则3: 如果overwrite为False, 则跳过已经存在的结果
    if not overwrite and setting_key in exist_results:
        return None

    return setting_key


def generate_tasks(num_tasks):
    task_name_to_cls = get_eval_tasks()

    # 排除部分不测试的数据集
    task_names = [
        task for task in task_name_to_cls.keys() if task not in SKIP_EVAL_TASKS
    ]

    # chunking策略列表
    strategies = [
        # "semantic_llama_index",
        "semantic_langchain",
        "fixed_token",
        # "fixed_text",
        # "recursive_chunking",
        # "sentences",
        "late_chunking",
    ]

    # chunk size
    chunk_size_list = [128, 256, 512, 1024]

    # model
    models = [
        "jinaai/jina-embeddings-v2-base-zh",
        "jinaai/jina-embeddings-v3",
        "BAAI/bge-m3",
        # "maidalun1020/bce-embedding-base_v1",
    ]

    # 笛卡尔集
    param_names = ["task", "strategy", "chunk_size", "model_name"]
    combinations = itertools.product(task_names, strategies, chunk_size_list, models)
    combinations = [dict(zip(param_names, combo)) for combo in combinations]
    # 去除非法和已测试的组合
    evaluated_key, benchmark = load_existing_results()
    eval_settings = set()
    for combo in combinations:
        valid_setting = get_valid_setting_str(combo, evaluated_key, OVERWRITE)
        if valid_setting is not None:
            eval_settings.add(valid_setting)

    return list(eval_settings), benchmark


def main(num_tasks):
    tasks, benchmark = generate_tasks(num_tasks)
    result_tasks = []

    for task_str in tasks:
        result = compute_task.delay(task_str)
        result_tasks.append({"task_id": task_str, "result": result})

    for result_task in result_tasks:
        result = result_task["result"].get()
        if result is None:
            continue
        benchmark.append(result)

        print(f"Task: {result_task['task_id']} finished.")

        with open(f"{OUTPUT_DIR}/benchmark.json", "w", encoding="utf-8") as f:
            json.dump(benchmark, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main(num_tasks=10)
