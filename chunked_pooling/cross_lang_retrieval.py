from collections import defaultdict
from datasets import load_dataset, DatasetDict

from mteb import AbsTaskRetrieval
from mteb.tasks import load_retrieval_data
from mteb.abstasks.TaskMetadata import TaskMetadata


class CrosslingualRetrievalBooksEn2Zh(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CrosslingualRetrievalBooksEn2Zh",
        description="",
        reference="https://None",
        dataset={
            "path": "maidalun1020/CrosslingualRetrievalBooksEn2Zh",
            "revision": "",
            "qrel_revision": "",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["en-zh"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
    )

    @property
    def description(self):
        return {
            "name": "CrosslingualRetrievalBooksEn2Zh",
            "hf_hub_name": "maidalun1020/CrosslingualRetrievalBooksEn2Zh",
            "reference": "",
            "description": "",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["en-zh"],
            "main_score": "ndcg_at_3",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["dataset"]["path"],
            self.metadata_dict["dataset"]["revision"],
            self.metadata_dict["dataset"]["qrel_revision"],
            self.metadata_dict["eval_splits"],
        )
        self.data_loaded = True


class CrosslingualRetrievalBooksZh2En(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CrosslingualRetrievalBooksZh2En",
        description="",
        reference="https://None",
        dataset={
            "path": "maidalun1020/CrosslingualRetrievalBooksZh2En",
            "revision": "",
            "qrel_revision": "",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["zh-en"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
    )

    @property
    def description(self):
        return {
            "name": "CrosslingualRetrievalBooksZh2En",
            "hf_hub_name": "maidalun1020/CrosslingualRetrievalBooksZh2En",
            "reference": "",
            "description": "",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["zh-en"],
            "main_score": "ndcg_at_3",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["dataset"]["path"],
            self.metadata_dict["dataset"]["revision"],
            self.metadata_dict["dataset"]["qrel_revision"],
            self.metadata_dict["eval_splits"],
        )
        self.data_loaded = True


class CrosslingualRetrievalFinanceEn2Zh(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CrosslingualRetrievalFinanceEn2Zh",
        description="",
        reference="https://None",
        dataset={
            "path": "maidalun1020/CrosslingualRetrievalFinanceEn2Zh",
            "revision": "",
            "qrel_revision": "",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["en-zh"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
    )

    @property
    def description(self):
        return {
            "name": "CrosslingualRetrievalFinanceEn2Zh",
            "hf_hub_name": "maidalun1020/CrosslingualRetrievalFinanceEn2Zh",
            "reference": "",
            "description": "",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["en-zh"],
            "main_score": "ndcg_at_3",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["dataset"]["path"],
            self.metadata_dict["dataset"]["revision"],
            self.metadata_dict["dataset"]["qrel_revision"],
            self.metadata_dict["eval_splits"],
        )
        self.data_loaded = True


class CrosslingualRetrievalFinanceZh2En(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CrosslingualRetrievalFinanceZh2En",
        description="",
        reference="https://None",
        dataset={
            "path": "maidalun1020/CrosslingualRetrievalFinanceZh2En",
            "revision": "",
            "qrel_revision": "",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["zh-en"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
    )

    @property
    def description(self):
        return {
            "name": "CrosslingualRetrievalFinanceZh2En",
            "hf_hub_name": "maidalun1020/CrosslingualRetrievalFinanceZh2En",
            "reference": "",
            "description": "",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["zh-en"],
            "main_score": "ndcg_at_3",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["dataset"]["path"],
            self.metadata_dict["dataset"]["revision"],
            self.metadata_dict["dataset"]["qrel_revision"],
            self.metadata_dict["eval_splits"],
        )
        self.data_loaded = True


class CrosslingualRetrievalLawEn2Zh(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CrosslingualRetrievalLawEn2Zh",
        description="",
        reference="https://None",
        dataset={
            "path": "maidalun1020/CrosslingualRetrievalLawEn2Zh",
            "revision": "",
            "qrel_revision": "",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["en-zh"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
    )

    @property
    def description(self):
        return {
            "name": "CrosslingualRetrievalLawEn2Zh",
            "hf_hub_name": "maidalun1020/CrosslingualRetrievalLawEn2Zh",
            "reference": "",
            "description": "",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["en-zh"],
            "main_score": "ndcg_at_3",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["dataset"]["path"],
            self.metadata_dict["dataset"]["revision"],
            self.metadata_dict["dataset"]["qrel_revision"],
            self.metadata_dict["eval_splits"],
        )
        self.data_loaded = True


class CrosslingualRetrievalLawZh2En(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CrosslingualRetrievalLawZh2En",
        description="",
        reference="https://None",
        dataset={
            "path": "maidalun1020/CrosslingualRetrievalLawZh2En",
            "revision": "",
            "qrel_revision": "",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["zh-en"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
    )

    @property
    def description(self):
        return {
            "name": "CrosslingualRetrievalLawZh2En",
            "hf_hub_name": "maidalun1020/CrosslingualRetrievalLawZh2En",
            "reference": "",
            "description": "",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["zh-en"],
            "main_score": "ndcg_at_3",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["dataset"]["path"],
            self.metadata_dict["dataset"]["revision"],
            self.metadata_dict["dataset"]["qrel_revision"],
            self.metadata_dict["eval_splits"],
        )
        self.data_loaded = True


class CrosslingualRetrievalPaperEn2Zh(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CrosslingualRetrievalPaperEn2Zh",
        description="",
        reference="https://None",
        dataset={
            "path": "maidalun1020/CrosslingualRetrievalPaperEn2Zh",
            "revision": "",
            "qrel_revision": "",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["en-zh"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
    )

    @property
    def description(self):
        return {
            "name": "CrosslingualRetrievalPaperEn2Zh",
            "hf_hub_name": "maidalun1020/CrosslingualRetrievalPaperEn2Zh",
            "reference": "",
            "description": "",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["en-zh"],
            "main_score": "ndcg_at_3",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["dataset"]["path"],
            self.metadata_dict["dataset"]["revision"],
            self.metadata_dict["dataset"]["qrel_revision"],
            self.metadata_dict["eval_splits"],
        )
        self.data_loaded = True


class CrosslingualRetrievalPaperZh2En(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CrosslingualRetrievalPaperZh2En",
        description="",
        reference="https://None",
        dataset={
            "path": "maidalun1020/CrosslingualRetrievalPaperZh2En",
            "revision": "",
            "qrel_revision": "",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["zh-en"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
    )

    @property
    def description(self):
        return {
            "name": "CrosslingualRetrievalPaperZh2En",
            "hf_hub_name": "maidalun1020/CrosslingualRetrievalPaperZh2En",
            "reference": "",
            "description": "",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["zh-en"],
            "main_score": "ndcg_at_3",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["dataset"]["path"],
            self.metadata_dict["dataset"]["revision"],
            self.metadata_dict["dataset"]["qrel_revision"],
            self.metadata_dict["eval_splits"],
        )
        self.data_loaded = True


class CrosslingualRetrievalWikiEn2Zh(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CrosslingualRetrievalWikiEn2Zh",
        description="",
        reference="https://None",
        dataset={
            "path": "maidalun1020/CrosslingualRetrievalWikiEn2Zh",
            "revision": "",
            "qrel_revision": "",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["en-zh"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
    )

    @property
    def description(self):
        return {
            "name": "CrosslingualRetrievalWikiEn2Zh",
            "hf_hub_name": "maidalun1020/CrosslingualRetrievalWikiEn2Zh",
            "reference": "",
            "description": "",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["en-zh"],
            "main_score": "ndcg_at_3",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["dataset"]["path"],
            self.metadata_dict["dataset"]["revision"],
            self.metadata_dict["dataset"]["qrel_revision"],
            self.metadata_dict["eval_splits"],
        )
        self.data_loaded = True


class CrosslingualRetrievalWikiZh2En(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CrosslingualRetrievalWikiZh2En",
        description="",
        reference="https://None",
        dataset={
            "path": "maidalun1020/CrosslingualRetrievalWikiZh2En",
            "revision": "",
            "qrel_revision": "",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["zh-en"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
    )

    @property
    def description(self):
        return {
            "name": "CrosslingualRetrievalWikiZh2En",
            "hf_hub_name": "maidalun1020/CrosslingualRetrievalWikiZh2En",
            "reference": "",
            "description": "",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["zh-en"],
            "main_score": "ndcg_at_3",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["dataset"]["path"],
            self.metadata_dict["dataset"]["revision"],
            self.metadata_dict["dataset"]["qrel_revision"],
            self.metadata_dict["eval_splits"],
        )
        self.data_loaded = True


class CrosslingualRetrievalQasEn2Zh(AbsTaskRetrieval):
    metadata = TaskMetadata(
        name="CrosslingualRetrievalQasEn2Zh",
        description="",
        reference="https://None",
        dataset={
            "path": "maidalun1020/CrosslingualRetrievalQasEn2Zh",
            "revision": "",
            "qrel_revision": "",
        },
        type="Retrieval",
        category="s2p",
        modalities=["text"],
        eval_splits=["dev"],
        eval_langs=["en-zh"],
        main_score="ndcg_at_10",
        date=None,
        domains=None,
        task_subtypes=None,
        license=None,
        annotations_creators=None,
        dialect=None,
        sample_creation=None,
        bibtex_citation=None,
    )

    @property
    def description(self):
        return {
            "name": "CrosslingualRetrievalQasEn2Zh",
            "hf_hub_name": "maidalun1020/CrosslingualRetrievalQasEn2Zh",
            "reference": "",
            "description": "",
            "type": "Retrieval",
            "category": "s2p",
            "eval_splits": ["dev"],
            "eval_langs": ["en-zh"],
            "main_score": "ndcg_at_3",
        }

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        self.corpus, self.queries, self.relevant_docs = load_retrieval_data(
            self.metadata_dict["dataset"]["path"],
            self.metadata_dict["dataset"]["revision"],
            self.metadata_dict["dataset"]["qrel_revision"],
            self.metadata_dict["eval_splits"],
        )
        self.data_loaded = True
