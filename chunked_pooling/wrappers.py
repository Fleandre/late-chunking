from typing import List, Optional, Union

import torch
import torch.nn as nn
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPooling
from FlagEmbedding import BGEM3FlagModel


def construct_document(doc):
    if isinstance(doc, str):
        return doc
    elif "title" in doc:
        return f'{doc["title"]} {doc["text"].strip()}'
    else:
        return doc["text"].strip()


class JinaEmbeddingsV3Wrapper(nn.Module):
    def __init__(
        self, model_name, tasks=["retrieval.query", "retrieval.passage"], **model_kwargs
    ):
        super().__init__()
        self._model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, **model_kwargs
        )
        self.tasks = tasks

    def encode_queries(
        self,
        sentences: Union[str, List[str]],
        *args,
        task: Optional[str] = None,
        **kwargs,
    ):
        return self._model.encode(sentences, *args, task=self.tasks[0], **kwargs)

    def encode_corpus(
        self,
        sentences: Union[str, List[str]],
        *args,
        **kwargs,
    ):
        _sentences = [construct_document(sentence) for sentence in sentences]
        return self._model.encode(_sentences, *args, task=self.tasks[1], **kwargs)

    def get_instructions(self):
        return [self._model._task_instructions[x] for x in self.tasks]

    def forward(self, *args, **kwargs):
        task_id = self._model._adaptation_map[self.tasks[1]]
        # 获取"input"字段以兼容sentence-trasformer
        num_examples = kwargs["input"]["input_ids"].shape[0]
        adapter_mask = torch.full(
            (num_examples,), task_id, dtype=torch.int32, device=self._model.device
        )
        model_outputs = self._model.forward(
            *args, adapter_mask=adapter_mask, **kwargs["input"]
        )
        return {"token_embeddings": model_outputs["last_hidden_state"]}

    @property
    def device(self):
        return self._model.device

    @staticmethod
    def has_instructions():
        return True


class NomicAIWrapper(nn.Module):
    def __init__(self, model_name, **model_kwargs):
        super().__init__()
        self._model = SentenceTransformer(
            model_name, trust_remote_code=True, **model_kwargs
        )
        self.instructions = ["search_query: ", "search_document: "]

    def get_instructions(self):
        return self.instructions

    def forward(self, *args, **kwargs):
        model_output = self._model.forward(kwargs)
        base_model_output = BaseModelOutputWithPooling(
            last_hidden_state=model_output["token_embeddings"],
            pooler_output=model_output["sentence_embedding"],
            attentions=model_output["attention_mask"],
        )
        return base_model_output

    def encode_queries(
        self,
        sentences: Union[str, List[str]],
        *args,
        **kwargs,
    ):
        return self._model.encode(
            [self.instructions[0] + s for s in sentences], *args, **kwargs
        )

    def encode_corpus(
        self,
        sentences: Union[str, List[str]],
        *args,
        **kwargs,
    ):
        return self._model.encode(
            [self.instructions[1] + construct_document(s) for s in sentences],
            *args,
            **kwargs,
        )

    @property
    def device(self):
        return self._model.device

    @staticmethod
    def has_instructions():
        return True

class BAAIBGEM3Wrapper(nn.Module):
    def __init__(
        self, model_name, **model_kwargs
    ):
        super().__init__()
        chunking_method = model_kwargs.pop("chunking_method")
        self.return_colbert = chunking_method == "late_chunking"
        self._model = BGEM3FlagModel(model_name, **model_kwargs, 
                                     normalize_embeddings=True,
                                     use_fp16=False)

    def encode(
        self,
        sentences: Union[str, List[str]],
        *args,
        **kwargs,
    ):
        kwargs.pop("output_value", None)
        embeddings = self._model.encode(
            sentences, *args, **kwargs,
            return_dense=True, return_colbert_vecs=self.return_colbert)
        
        if self.return_colbert:
            token_embeds = [torch.from_numpy(embed) for embed in embeddings["colbert_vecs"]]
            return token_embeds
        
        return torch.tensor(embeddings["dense_vecs"])

    def encode_queries(
        self,
        sentences: Union[str, List[str]],
        *args,
        **kwargs,
    ):
        embeds = self.encode(sentences, *args, **kwargs)
        if self.return_colbert:
            mean_pooled = np.array([embed.numpy().mean(axis=0) for embed in embeds])
            normalized = mean_pooled / np.linalg.norm(mean_pooled, axis=1)[:, None]
            return normalized
        return embeds

    def encode_corpus(
        self,
        sentences: Union[str, List[str]],
        *args,
        **kwargs,
    ):
        return self.encode(sentences, *args, **kwargs)

    @property
    def device(self):
        return self._model.device

    @staticmethod
    def has_instructions():
        return False

MODEL_WRAPPERS = {
    "jinaai/jina-embeddings-v3": JinaEmbeddingsV3Wrapper,
    "sentence-transformers/all-MiniLM-L6-v2": SentenceTransformer,
    "nomic-ai/nomic-embed-text-v1": NomicAIWrapper,
    "BAAI/bge-m3": BAAIBGEM3Wrapper,
}

MODELS_WITHOUT_PROMPT_NAME_ARG = [
    "jinaai/jina-embeddings-v2-small-en",
    "jinaai/jina-embeddings-v2-base-en",
    "jinaai/jina-embeddings-v2-base-zh",
    "jinaai/jina-embeddings-v3",
]


def remove_unsupported_kwargs(original_encode):
    def wrapper(self, *args, **kwargs):
        # Remove 'prompt_name' from kwargs if present
        kwargs.pop("prompt_name", None)
        kwargs.pop("request_qid", None)
        return original_encode(self, *args, **kwargs)

    return wrapper


def load_model(model_name, **model_kwargs):
    if model_name in MODEL_WRAPPERS:
        model = MODEL_WRAPPERS[model_name](model_name, **model_kwargs)
        if hasattr(MODEL_WRAPPERS[model_name], "has_instructions"):
            has_instructions = MODEL_WRAPPERS[model_name].has_instructions()
        else:
            has_instructions = False
    else:
        # model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model = SentenceTransformer(model_name, trust_remote_code=True)
        has_instructions = False

    # encode functions of various models do not support all sentence transformers kwargs parameter
    if model_name in MODELS_WITHOUT_PROMPT_NAME_ARG:
        ENCODE_FUNC_NAMES = ["encode", "encode_queries", "encode_corpus"]
        for func_name in ENCODE_FUNC_NAMES:
            if hasattr(model, func_name):
                setattr(
                    model,
                    func_name,
                    remove_unsupported_kwargs(getattr(model, func_name)),
                )

    return model, has_instructions
