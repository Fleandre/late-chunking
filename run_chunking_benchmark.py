import json
import torch.cuda
from mteb import MTEB
from transformers import AutoModel, AutoTokenizer
import itertools
import multiprocessing

from chunked_pooling.chunked_eval_tasks import *
from chunked_pooling.wrappers import load_model

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


def run_eval(eval_setting_str, task_name_to_cls, batch_size, benchmark, return_dict):
    eval_setting = json.loads(eval_setting_str)
    print(f"Evaluation Setting: {eval_setting}")
    task_cls = task_name_to_cls[eval_setting["task"]]
    model_name = eval_setting["model_name"]
    strategy = eval_setting["strategy"]
    chunk_size = eval_setting["chunk_size"]
    n_sentences = 5
    chunking_model = None
    truncate_max_length = None

    # build dependencies
    model, has_instructions = load_model(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    chunking_args = {
        "chunk_size": chunk_size,
        "n_sentences": n_sentences,
        "chunking_strategy": strategy,
        "model_has_instructions": has_instructions,
        "embedding_model_name": (chunking_model if chunking_model else model_name),
    }

    tasks = [
        task_cls(
            tokenizer=tokenizer,
            prune_size=None,
            truncate_max_length=truncate_max_length,
            **chunking_args,
        )
    ]

    # Evaluate
    evaluation = MTEB(
        tasks=tasks,
        tokenizer=tokenizer,
        prune_size=None,
        **chunking_args,
    )
    try:
        res = evaluation.run(
            model,
            output_folder="results",
            eval_splits=tasks[0].eval_splits,
            overwrite_results=True,
            encode_kwargs={"batch_size": batch_size},
            verbosity=0,
        )
    except:
        print(f"Batch size {batch_size} is too large. Reduce batch size and try again.")
        return_dict[eval_setting_str] = (-1, None)
        return return_dict

    # 保存结果
    res_dict = res[0].to_dict()
    res_dict["model_name"] = model_name
    res_dict["chunking_strategy"] = strategy
    res_dict["chunk_size"] = chunk_size
    res_dict["n_sentences"] = n_sentences
    res_dict["chunking_model"] = chunking_model

    return_dict[eval_setting_str] = (batch_size, res_dict)

    return return_dict


def main():
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

    # 定义尝试的batch size列表，从大到小
    batch_sizes = [32, 16, 8, 4, 2, 1]
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    optimal_batch_size_dict = dict()
    for i, eval_setting in enumerate(eval_settings):
        for batch_size in batch_sizes:
            p = multiprocessing.Process(
                target=run_eval,
                args=(
                    eval_setting,
                    task_name_to_cls,
                    batch_size,
                    benchmark,
                    return_dict,
                ),
            )
            p.start()
            p.join()

            # 检查是否成功运行
            optimal_batch_size, res_dict = return_dict[eval_setting]
            optimal_batch_size_dict[eval_setting] = optimal_batch_size
            if optimal_batch_size != -1:
                print(
                    f"Parameters: {eval_setting}, Optimal batch size: {optimal_batch_size}"
                )
                # 保存结果
                benchmark.append(res_dict)
                with open(f"{OUTPUT_DIR}/benchmark.json", "w", encoding="utf-8") as f:
                    json.dump(benchmark, f, ensure_ascii=False, indent=4)
                break

        with open(f"{OUTPUT_DIR}/batch_size.json", "w", encoding="utf-8") as f:
            json.dump(optimal_batch_size_dict, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
