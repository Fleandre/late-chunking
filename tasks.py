import json
import torch.cuda
from mteb import MTEB
from transformers import AutoModel, AutoTokenizer
import multiprocessing
from celery import Celery

from chunked_pooling.chunked_eval_tasks import *
from chunked_pooling.wrappers import load_model

# 定义 Celery 实例，使用 Redis 作为消息队列
app = Celery(
    "tasks",
    broker="redis://sh-ppml-01.alipay.net:6379/0",
    backend="redis://sh-ppml-01.alipay.net:6379/0",
)

task_name_to_cls = get_eval_tasks()


def run_eval(eval_setting_str, task_name_to_cls, batch_size, return_dict):
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


@app.task
def compute_task(eval_setting):
    # 定义尝试的batch size列表，从大到小
    batch_sizes = [32, 16, 8, 4, 2, 1]
    manager = multiprocessing.Manager()
    sub_proc_return_dict = manager.dict()
    optimal_batch_size_dict = dict()
    for batch_size in batch_sizes:
        p = multiprocessing.Process(
            target=run_eval,
            args=(
                eval_setting,
                task_name_to_cls,
                batch_size,
                sub_proc_return_dict,
            ),
        )
        p.start()
        p.join()

        # 检查是否成功运行
        optimal_batch_size, res_dict = sub_proc_return_dict[eval_setting]
        optimal_batch_size_dict[eval_setting] = optimal_batch_size
        if optimal_batch_size != -1:
            print(
                f"Parameters: {eval_setting}, Optimal batch size: {optimal_batch_size}"
            )
            return res_dict
    return None
