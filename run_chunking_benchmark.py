import click
import json
import torch.cuda
from mteb import MTEB
from transformers import AutoModel, AutoTokenizer
import itertools

from chunked_pooling.chunked_eval_tasks import *
from chunked_pooling.wrappers import load_model

BATCH_SIZE = 1
SKIP_EVAL_TASKS = [
    "ClimateFEVERChunked",
    "DBPediaChunked",
    "FEVERChunked",
    "HotpotQAChunked",
    "MSMARCOChunked",
]
OUTPUT_DIR = "results"


# 定义不合法组合的规则
def get_valid_setting_str(eval_setting):
    task_cls = eval_setting["task"]
    model_name = eval_setting["model_name"]
    strategy = eval_setting["strategy"]
    chunk_size = eval_setting["chunk_size"]

    # 规则1: semantic chunking策略下chunk size无影响
    if "semantic" in strategy:
        eval_setting["chunk_size"] = -1

    # 规则2: bce-embedding-base_v1最长上下文为512, 因此无法使用late_chunking
    if "bce-embedding-base_v1" in model_name and strategy == "late_chunking":
        return None

    return json.dumps(eval_setting, sort_keys=True)


def main():
    benchmark = []
    task_name_to_cls = get_eval_tasks()

    # 排除部分不测试的数据集
    task_names = [
        task for task in task_name_to_cls.keys() if task not in SKIP_EVAL_TASKS
    ]

    # chunking策略列表
    strategies = [
        "semantic_llama_index",
        "semantic_langchain",
        "fixed_token",
        "fixed_text",
        # "recursive_chunking",
        # "sentences",
        "late_chunking",
    ]

    # chunk size
    chunk_size_list = [128, 256, 512, 1024, 1536, 2048]

    # model
    models = [
        "jinaai/jina-embeddings-v2-base-zh",
        "jinaai/jina-embeddings-v3",
        "BAAI/bge-m3",
        "maidalun1020/bce-embedding-base_v1",
    ]

    # 笛卡尔集
    param_names = ["task", "strategy", "chunk_size", "model_name"]
    combinations = itertools.product(task_names, strategies, chunk_size_list, models)
    eval_settings = [dict(zip(param_names, combo)) for combo in combinations]
    # 去除非法和重复的组合
    eval_settings = set(
        get_valid_setting_str(eval_setting) for eval_setting in eval_settings
    )
    # json转换为dict
    eval_settings = [
        json.loads(eval_setting)
        for eval_setting in eval_settings
        if eval_setting is not None
    ]

    for eval_setting in eval_settings:
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
        res = evaluation.run(
            model,
            output_folder="results",
            eval_splits=tasks[0].eval_splits,
            overwrite_results=True,
            encode_kwargs={"batch_size": BATCH_SIZE},
            verbosity=2,
        )

        # 保存结果
        res_dict = res[0].to_dict()
        res_dict["model_name"] = model_name
        res_dict["chunking_strategy"] = strategy
        res_dict["chunk_size"] = chunk_size
        res_dict["n_sentences"] = n_sentences
        res_dict["chunking_model"] = chunking_model

        benchmark.append(res_dict)
        with open(f"{OUTPUT_DIR}/benchmark.json", "w", encoding="utf-8") as f:
            json.dump(benchmark, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
