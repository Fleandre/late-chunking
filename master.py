import json
import time
import itertools
from tasks import compute_task
from chunked_pooling.chunked_eval_tasks import *
from eval_utils import *


def generate_tasks():
    task_name_to_cls = get_eval_tasks()

    # 排除部分不测试的数据集
    task_names = filter_tasks(task_name_to_cls)

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


def main():
    tasks, benchmark = generate_tasks()
    result_tasks = []

    for task_str in tasks:
        result = compute_task.delay(task_str)
        result_tasks.append({"task_id": task_str, "result": result})

    results = []
    while result_tasks:
        for result_task in result_tasks:
            if not result_task["result"].ready():
                continue
            # 任务已完成
            result = result_task["result"].get()
            if result is None:
                continue

            benchmark.append(result)

            print(f"Task: {result_task['task_id']} finished.")

            with open(f"{OUTPUT_DIR}/benchmark.json", "w", encoding="utf-8") as f:
                json.dump(benchmark, f, ensure_ascii=False, indent=4)

            # 移除已完成任务
            result_tasks.remove(result_task)
        time.sleep(30)  # 轮询间隔时间


if __name__ == "__main__":
    main()
