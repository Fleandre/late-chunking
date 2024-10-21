import click
import torch.cuda
from mteb import MTEB
from transformers import AutoModel, AutoTokenizer

from chunked_pooling.chunked_eval_tasks import *
from chunked_pooling.wrappers import load_model

DEFAULT_CHUNKING_STRATEGY = "fixed_token"
DEFAULT_CHUNK_SIZE = 256
DEFAULT_N_SENTENCES = 5
BATCH_SIZE = 16
OUTPUT_DIR = "results"


@click.command()
@click.option(
    "--model-name",
    default="jinaai/jina-embeddings-v2-small-en",
    help="The name of the model to use.",
)
@click.option(
    "--strategy",
    default=DEFAULT_CHUNKING_STRATEGY,
    help="The chunking strategy to be applied.",
)
@click.option(
    "--task-name", default="SciFactChunked", help="The evaluation task to perform."
)
@click.option(
    "--eval-split", default="None", help="The name of the evaluation split in the task."
)
@click.option(
    "--chunking-model",
    default=None,
    required=False,
    help="The name of the model used for semantic chunking.",
)
@click.option(
    "--truncate-max-length",
    default=None,
    type=int,
    help="Maximum number of tokens; By default, no truncation is done.",
)
@click.option(
    "--chunk-size",
    default=DEFAULT_CHUNK_SIZE,
    type=int,
    help="Number of tokens per chunk for fixed strategy.",
)
@click.option(
    "--n-sentences",
    default=DEFAULT_N_SENTENCES,
    type=int,
    help="Number of sentences per chunk for sentence strategy.",
)
def main(
    model_name,
    strategy,
    task_name,
    eval_split,
    chunking_model,
    truncate_max_length,
    chunk_size,
    n_sentences,
):
    task_clses = get_eval_tasks()

    assert task_name in task_clses, f"Unknown task name: {task_name}"
    task_cls = task_clses[task_name]

    model, has_instructions = load_model(model_name, chunking_method=strategy)

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

    # Evaluate
    tasks = [
        task_cls(
            tokenizer=tokenizer,
            prune_size=None,
            truncate_max_length=truncate_max_length,
            **chunking_args,
        )
    ]

    eval_splits = tasks[0].eval_splits if eval_split == "None" else [eval_split]

    evaluation = MTEB(
        tasks=tasks,
        tokenizer=tokenizer,
        prune_size=None,
        **chunking_args,
    )
    res = evaluation.run(
        model,
        output_folder=OUTPUT_DIR,
        eval_splits=eval_splits,
        overwrite_results=True,
        encode_kwargs={"batch_size": BATCH_SIZE},
        verbosity=2,
    )
    print(res[0].scores)
    pass


if __name__ == "__main__":
    main()
