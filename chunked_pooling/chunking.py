import bisect
import logging
from typing import Dict, List, Optional, Tuple, Union

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer

# Set the logging level to WARNING to suppress INFO and DEBUG messages
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

CHUNKING_STRATEGIES = [
    "semantic",
    "fixed_token",
    "fixed_text",
    "sentences",
    "late_chunking",
]


class Chunker:
    def __init__(
        self,
        chunking_strategy: str,
    ):
        if chunking_strategy not in CHUNKING_STRATEGIES:
            raise ValueError("Unsupported chunking strategy: ", chunking_strategy)
        self.chunking_strategy = chunking_strategy
        self.embed_model = None
        self.embedding_model_name = None

    def _setup_semantic_chunking(self, embedding_model_name):
        if embedding_model_name:
            self.embedding_model_name = embedding_model_name

        self.embed_model = HuggingFaceEmbedding(
            model_name=self.embedding_model_name,
            trust_remote_code=True,
            embed_batch_size=1,
        )
        self.splitter = SemanticSplitterNodeParser(
            embed_model=self.embed_model,
            show_progress=False,
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

    def chunk_semantically(
        self,
        text: str,
        tokenizer: "AutoTokenizer",
        use_token_index: bool,
        embedding_model_name: Optional[str] = None,
    ) -> List[Tuple[int, int]]:
        if self.embed_model is None:
            self._setup_semantic_chunking(embedding_model_name)

        # Get semantic nodes
        nodes = [
            (node.start_char_idx, node.end_char_idx)
            for node in self.splitter.get_nodes_from_documents(
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
        if chunking_strategy == "semantic":
            return self.chunk_semantically(
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
