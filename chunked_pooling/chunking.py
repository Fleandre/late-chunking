import bisect
import logging
from typing import Dict, List, Optional, Tuple, Union

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding as llama_hf_embedding,
)
from transformers import AutoTokenizer
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings as langchian_hf_embedding

# Set the logging level to WARNING to suppress INFO and DEBUG messages
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

CHUNKING_STRATEGIES = [
    "semantic_llama_index",
    "semantic_langchain",
    "fixed_token",
    "fixed_text",
    "recursive_chunking",
    "sentences",
    "late_chunking",
]


class Chunker:
    def __init__(self, chunking_strategy: str, truncate_max_length: None):
        if chunking_strategy not in CHUNKING_STRATEGIES:
            raise ValueError("Unsupported chunking strategy: ", chunking_strategy)
        self.chunking_strategy = chunking_strategy
        self.llama_embed_model = None
        self.langchain_embed_model = None
        self.embedding_model_name = None
        self.truncate_max_length = truncate_max_length

    def _setup_semantic_chunking(self, embedding_model_name):
        if embedding_model_name:
            self.embedding_model_name = embedding_model_name

        self.llama_embed_model = llama_hf_embedding(
            model_name=self.embedding_model_name,
            trust_remote_code=True,
            embed_batch_size=1,
        )

        self.llama_splitter = SemanticSplitterNodeParser(
            embed_model=self.llama_embed_model,
            show_progress=False,
        )

        self.langchain_embed_model = langchian_hf_embedding(
            model_name=self.embedding_model_name,
            model_kwargs={"trust_remote_code": True},
        )
        self.langchain_splitter = SemanticChunker(
            self.langchain_embed_model,
            breakpoint_threshold_type="percentile",
            sentence_split_regex=r"(?<=[.?!，？！。；])\s*",
        )

    def _spans_by_char_to_by_token(self, text, tokenizer, spans_by_char):
        # Tokenize the entire text
        tokens = tokenizer.encode_plus(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            padding=True,
            truncation=True,
        )
        token_offsets = tokens.offset_mapping

        # 转换成token的index
        chunk_spans_by_token = []

        for char_start, char_end in spans_by_char:
            # Convert char indices to token indices
            start_chunk_index = bisect.bisect_left(
                [offset[0] for offset in token_offsets], char_start
            )
            end_chunk_index = bisect.bisect_right(
                [offset[1] for offset in token_offsets], char_end
            )

            # Add the chunk span if it's within the tokenized text
            if start_chunk_index < len(token_offsets) and end_chunk_index <= len(
                token_offsets
            ):
                chunk_spans_by_token.append((start_chunk_index, end_chunk_index))
            else:
                break

        return chunk_spans_by_token

    def _spans_by_token_to_by_char(self, token_offsets, spans_by_token):
        chunk_spans_by_char = []
        for start, end in spans_by_token:
            chunk_spans_by_char.append(
                (token_offsets[start][0], token_offsets[end - 1][1])
            )
        return chunk_spans_by_char

    def chunk_semantically_with_llamaindex(
        self,
        text: str,
        tokenizer: "AutoTokenizer",
        use_token_index: bool,
        embedding_model_name: Optional[str] = None,
    ) -> List[Tuple[int, int]]:
        if self.llama_embed_model is None:
            self._setup_semantic_chunking(embedding_model_name)

        # Get semantic nodes
        nodes = [
            (node.start_char_idx, node.end_char_idx)
            for node in self.llama_splitter.get_nodes_from_documents(
                [Document(text=text)], show_progress=False
            )
        ]

        chunk_spans_by_char = []
        for char_start, char_end in nodes:
            chunk_spans_by_char.append((char_start, char_end))

        # 返回以原始text中char index记录的chunk坐标
        if not use_token_index:
            return chunk_spans_by_char

        return self._spans_by_char_to_by_token(text, tokenizer, chunk_spans_by_char)

    def chunk_semantically_with_langchain(
        self,
        text: str,
        tokenizer: "AutoTokenizer",
        use_token_index: bool,
        embedding_model_name: Optional[str] = None,
    ) -> List[Tuple[int, int]]:
        if self.langchain_embed_model is None:
            self._setup_semantic_chunking(embedding_model_name)

        def recursive_split(to_split):
            # 递归切分文本，直到所有文本块的token数都小于阈值
            text_chunks = self.langchain_splitter.split_text(to_split)
            if len(text_chunks) == 1:
                return [to_split]

            result_chunks = []

            for chunk in text_chunks:
                if (
                    len(tokenizer.encode(chunk, add_special_tokens=True))
                    > self.truncate_max_length
                ):

                    # 递归切分当前块
                    result_chunks.extend(recursive_split(chunk))
                else:
                    result_chunks.append(chunk)

            return result_chunks

        text_chunks = recursive_split(text)

        chunk_spans_by_char = []
        idx = 0
        for text_chunk in text_chunks:
            chunk_spans_by_char.append((idx, idx + len(text_chunk)))
            idx += len(text_chunk) + 1

        # 返回以原始text中char index记录的chunk坐标
        if not use_token_index:
            return chunk_spans_by_char

        return self._spans_by_char_to_by_token(text, tokenizer, chunk_spans_by_char)

    def chunk_by_tokens(
        self,
        text: str,
        chunk_size: int,
        tokenizer: "AutoTokenizer",
        use_token_index: bool,
    ) -> List[Tuple[int, int, int]]:
        tokens = tokenizer.encode_plus(
            text, return_offsets_mapping=True, add_special_tokens=False
        )
        token_offsets = tokens.offset_mapping

        chunk_spans_by_token = []
        for i in range(0, len(token_offsets), chunk_size):
            chunk_end = min(i + chunk_size, len(token_offsets))
            if chunk_end - i > 0:
                chunk_spans_by_token.append((i, chunk_end))

        if use_token_index:
            return chunk_spans_by_token

        return self._spans_by_token_to_by_char(token_offsets, chunk_spans_by_token)

    def chunk_by_text(
        self,
        text: str,
        chunk_size: int,
        tokenizer: "AutoTokenizer",
        use_token_index: bool,
    ) -> List[Tuple[int, int]]:
        chunk_spans_by_char = []
        for i in range(0, len(text), chunk_size):
            chunk_end = min(i + chunk_size, len(text))
            if chunk_end - i > 0:
                chunk_spans_by_char.append((i, chunk_end))

        if not use_token_index:
            return chunk_spans_by_char

        return self._spans_by_char_to_by_token(text, tokenizer, chunk_spans_by_char)

    def chunk_by_sentences(
        self,
        text: str,
        n_sentences: int,
        tokenizer: "AutoTokenizer",
        use_token_index: bool,
    ) -> List[Tuple[int, int, int]]:
        tokens = tokenizer.encode_plus(
            text, return_offsets_mapping=True, add_special_tokens=False
        )
        token_offsets = tokens.offset_mapping

        chunk_spans = []
        chunk_start = 0
        count_chunks = 0
        for i in range(0, len(token_offsets)):
            if tokens.tokens(0)[i] in (".", "!", "?") and (
                (len(tokens.tokens(0)) == i + 1)
                or (tokens.token_to_chars(i).end != tokens.token_to_chars(i + 1).start)
            ):
                count_chunks += 1
                if count_chunks == n_sentences:
                    chunk_spans.append((chunk_start, i + 1))
                    chunk_start = i + 1
                    count_chunks = 0
        if len(tokens.tokens(0)) - chunk_start > 1:
            chunk_spans.append((chunk_start, len(tokens.tokens(0))))

        if use_token_index:
            return chunk_spans

        return self._spans_by_token_to_by_char(token_offsets, chunk_spans)

    def chunk(
        self,
        text: str,
        tokenizer: "AutoTokenizer",
        chunking_strategy: str = None,
        chunk_size: Optional[int] = None,
        n_sentences: Optional[int] = None,
        embedding_model_name: Optional[str] = None,
        use_token_index: bool = True,
    ):
        chunking_strategy = chunking_strategy or self.chunking_strategy
        if chunking_strategy == "semantic_llama_index":
            return self.chunk_semantically_with_llamaindex(
                text,
                embedding_model_name=embedding_model_name,
                tokenizer=tokenizer,
                use_token_index=use_token_index,
            )
        elif chunking_strategy == "semantic_langchain":
            return self.chunk_semantically_with_langchain(
                text,
                embedding_model_name=embedding_model_name,
                tokenizer=tokenizer,
                use_token_index=use_token_index,
            )
        elif chunking_strategy == "fixed_token":
            if chunk_size < 10:
                raise ValueError("Chunk size must be greater than 10.")
            return self.chunk_by_tokens(
                text,
                chunk_size,
                tokenizer,
                use_token_index=use_token_index,
            )
        elif chunking_strategy == "fixed_text":
            if chunk_size < 10:
                raise ValueError("Chunk size must be greater than 10.")
            return self.chunk_by_text(
                text,
                chunk_size,
                tokenizer,
                use_token_index=use_token_index,
            )
        elif chunking_strategy == "sentences":
            return self.chunk_by_sentences(
                text,
                n_sentences,
                tokenizer,
                use_token_index=use_token_index,
            )
        else:
            raise ValueError("Unsupported chunking strategy")
