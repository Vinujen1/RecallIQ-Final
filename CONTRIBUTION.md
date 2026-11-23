# Contributions — RecallIQ

**Course:** ARTE 4110U — Deep Learning Architecture and Algorithms  

**Group:** 5  

**Repository:** https://github.com/Vinujen1/RecallIQ-Final  

**Academic term:** Fall 2026  

**Document scope:** Milestone 1 proposal and Milestone 2 technical checkpoint contributions

## 1. Team and consolidated Git history

| Commit | Recorded Git author / reported collaborator | Development milestone | Relevant files and artifacts |
|---|---|---|---|
| [`5152e8e`](https://github.com/Vinujen1/RecallIQ-Final/commit/5152e8e) | Vinujen Dilogen (`Vinujen1`) | Project initialization, recall dataset processing | `ml/recall_data/`, `docs/DATA_PIPELINE.md`, `tests/ml/test_recall_data.py` |
| [`fba6c3b`](https://github.com/Vinujen1/RecallIQ-Final/commit/fba6c3b) | Rukshan Baskaran (`rukshanbas342`) | BM25 and MiniLM retrieval baselines | `ml/retrieval/`, `docs/RETRIEVAL.md`, `tests/ml/test_retrieval.py` |
| [`fc669bc`](https://github.com/Vinujen1/RecallIQ-Final/commit/fc669bc) | Rukshan Baskaran (`rukshanbas342`) — Git author; Ayngaraan Chandramohan — reported collaborator (specific contribution pending confirmation) | PyTorch training, dataset splitting, evaluation and experiment artifacts | `ml/training/`, `ml/configs/`, `ml/evaluation/`, `data/reviews/`, `experiments/runs/`, `experiments/reports/` |
| [`9dec38c`](https://github.com/Vinujen1/RecallIQ-Final/commit/9dec38c) | Vinujen Dilogen (`Vinujen1`) | Evaluation improvements, Model Lab and frontend search integration | `ml/evaluation/`, `experiments/reports/seeds.md`, `frontend/src/app/`, `frontend/src/components/` |
| [`a6e1394`](https://github.com/Vinujen1/RecallIQ-Final/commit/a6e1394) | Vinujen Dilogen (`Vinujen1`) | Grounded RAG, LLM safeguards, fallback and integration tests | `ml/rag/`, `backend/app/llm_check.py`, `backend/app/routers/search.py`, `frontend/src/components/search/EvidenceSummary.tsx`, `tests/rag/`, `tests/backend/test_explain_llm.py`, `docs/RAG.md` |



## 2. Milestone 1 — Proposal and project planning

The following proposal-writing assignments were reported by the team and correspond to the headings in the submitted November 2, 2026 proposal.

| Team member | Proposal sections | Work described |
|---|---|---|
| **Rukshan Baskaran** | **Problem; Challenges** | Described difficulties retrieving product recalls from informal consumer descriptions, including limited query data, near-duplicate products, and exact identifier verification. |
| **Vinujen Dilogen** | **Data; Pre-Processing; Method & Baseline** | Described the Government of Canada recall dataset, filtering and splitting strategy, preparation of query–notice pairs, MiniLM fine-tuning, contrastive loss, hard negatives, and BM25/pretrained MiniLM baselines. |
| **Ayngaraan Chandramohan** | **Related Work; Reference List** | Prepared the related-work discussion and references concerning Sentence-BERT, MiniLM, dense retrieval, and information-retrieval benchmarks. |

These entries concern **who prepared the proposal sections**. They are distinct from the **planned implementation responsibilities** listed in the proposal. The author of the separate **Evaluation Plan** section has not been specified and is not assigned here.

## 3. Milestone 2 — Technical checkpoint: machine learning and RAG contributions

**Scope:** This section covers the ML core, retrieval baselines, PyTorch training, evaluation, experiments, and the evidence-grounded RAG system implemented for the checkpoint. Frontend work is included only where it directly presents retrieval results or RAG explanations.

| Member | ML and RAG work associated with the milestone | Key evidence / files | Commit |
|---|---|---|---|
| **Vinujen Dilogen** | Recall-data ingestion and preparation, including cleaning, filtering, grouping, sampling, and SQLite storage | `ml/recall_data/build.py`, `clean.py`, `grouping.py`, `load.py`, `sample.py`, `storage.py`; `docs/DATA_PIPELINE.md`; `tests/ml/test_recall_data.py` | `5152e8e` |
| **Rukshan Baskaran** | BM25 keyword retrieval, pretrained MiniLM embeddings, semantic retrieval and retrieval tests | `ml/retrieval/bm25.py`, `corpus.py`, `embed.py`, `semantic.py`, `search.py`; `docs/RETRIEVAL.md`; `tests/ml/test_retrieval.py` | `fba6c3b` |
| **Rukshan Baskaran** | Recorded training/evaluation milestone: MiniLM fine-tuning, contrastive training, hard-negative support, model configurations, training runs, checkpointing, and evaluation scripts | `ml/training/`, `ml/configs/`, `ml/evaluation/`, `experiments/runs/`, `experiments/reports/`, `docs/TRAINING.md`, `docs/EVALUATION.md` | `fc669bc` |
| **Ayngaraan Chandramohan** |  `ml/evaluation/splits.py`, `ml/evaluation/split_review\.py`, `data/reviews/`; **these are not yet verified as his work**. | `fc669bc` |
| **Vinujen Dilogen** | Subsequent changes to evaluation tooling, annotation workflows, multi-seed experiment reporting, interactive Model Lab visualizations and retrieval result presentation | `ml/evaluation/annotations.py`, `ml/evaluation/evaluate.py`, `experiments/reports/seeds.md`, `frontend/src/app/model-lab/page.tsx`, `frontend/src/components/lab/` | `9dec38c` |
| **Vinujen Dilogen** | Grounded retrieval-augmented generation: OpenAI-backed evidence summaries, citation/claim safeguards, deterministic fallback, API integration, live-check utility, UI explanation display and tests | `ml/rag/llm.py`, `ml/rag/guards.py`, `ml/rag/pipeline.py`, `backend/app/routers/search.py`, `backend/app/llm_check.py`, `frontend/src/components/search/EvidenceSummary.tsx`, `tests/rag/test_llm.py`, `tests/rag/test_guards.py`, `tests/backend/test_explain_llm.py`, `docs/RAG.md` | `a6e1394` |

**Experiment ownership still to verify:** The team must identify who personally ran and interpreted each training run, performed the split-review decisions, and reviewed the reported figures. A commit containing a log or results file does not by itself prove who ran the experiment. Assign these entries before final submission if relevant.

### 3.1 Machine learning experiments and preliminary results

The implementation compares four approaches:

1\. **BM25:** lexical retrieval baseline (no neural training).

2\. **Model A:** pretrained `all-MiniLM-L6-v2` semantic retrieval baseline (no recall-specific fine-tuning).

3\. **Model B:** MiniLM fine-tuned with Multiple Negatives Ranking Loss (in-batch negatives).

4\. **Model C:** MiniLM fine-tuned with additional hard negatives.

Training configurations, checkpoint-selection history, loss curves, and evaluation outputs are recorded under `ml/configs/`, `experiments/runs/`, and `experiments/reports/`. The preliminary proxy evaluation used **391 automatically generated validation queries** and reported Recall@5, MRR@10, and nDCG@10 for the four systems. Controlled experiments included comparison of in-batch versus hard-negative training and repeated runs with three seeds.

**Evaluation limitation:** These are generated-query proxy diagnostics, **not independently human-written and reviewed validation results**. They must be labelled accordingly in the checkpoint and not presented as final real-world performance.

### 3.2 RAG and identifier verification implementation (Vinujen; commit `a6e1394` and related integration)

The RAG portion is a substantial implementation contribution in addition to the ML training pipeline. The following functionality is present in the repository; the contributor must verify the personal work attribution before submission.

| Feature | Implemented functionality | Supporting files |
|---|---|---|
| **Retrieval-to-answer pipeline** | Accepts a consumer query, retrieves relevant official Canadian recall notices, and supplies at most three selected notices as evidence for an optional explanation. | `ml/rag/pipeline.py`, `backend/app/routers/search.py` |
| **LLM integration** | Configures a GPT-4o-mini-backed explanation path with structured output and evidence-linked recall identifiers; keeps ordinary search functional without the LLM. | `ml/rag/llm.py`, `backend/app/config.py`, `backend/app/llm_check.py` |
| **Grounding safeguards** | Checks generated claims against retrieved evidence, including recall IDs, product identifiers, names, numbers, quoted text and allowed official URLs; rejects unsupported or unsafe claims rather than presenting them as confirmed facts. | `ml/rag/guards.py`, `tests/rag/test_guards.py` |
| **Deterministic fallback and error handling** | Returns a template-based explanation if the provider is unavailable, times out, refuses, or produces an answer rejected by the guards; surfaces a fallback reason. | `ml/rag/pipeline.py`, `ml/rag/llm.py`, `tests/rag/test_llm.py`, `tests/backend/test_explain_llm.py` |
| **Exact product-identifier verification** | Integrates deterministic MATCH/MISMATCH/UNKNOWN checks for UPC, model, lot, and date-like identifiers to avoid treating semantic similarity as proof of a matching product. The verifier code was introduced in the training/evaluation milestone (`fc669bc`); it should not automatically be attributed exclusively to the RAG commit. | `ml/verification/identifiers.py`, `tests/verification/test_identifiers.py`, `ml/rag/pipeline.py` |
| **API and UI presentation** | Connects the explanation endpoint to the backend and renders evidence summaries, citation information, and fallback messaging in the search interface. The initial UI was introduced in `9dec38c`; further explanation changes occurred in `a6e1394`. | `backend/app/routers/search.py`, `backend/app/schemas.py`, `frontend/src/components/search/EvidenceSummary.tsx`, `frontend/src/components/search/SearchApp.tsx` |
| **Testing and documentation** | Adds LLM mocks and integration tests, a live-provider check utility, and detailed RAG design documentation. | `tests/rag/`, `tests/backend/test_explain_llm.py`, `backend/app/llm_check.py`, `docs/RAG.md` |

**Evaluation boundary:** Automated tests and a small number of live LLM checks demonstrate functionality and guard behavior, **not** independently measured answer accuracy, citation faithfulness, or a reliable overall acceptance rate. A larger human-reviewed RAG quality evaluation remains future work. RecallIQ is a retrieval aid and must not claim that a product is safe or definitively recalled based solely on model output.

## 4. Generative AI usage and provenance

The team used generative AI assistance during **system design and architecture planning**, including discussion of component responsibilities, data flow, alternative approaches, and integration decisions. All three team members are included below. The team reports that each member used Claude Code and ChatGPT. Each member should confirm the described purposes and disclose any additional uses before submission.

| Team member | Tool | Purpose / disclosure |
|---|---|---|
| **Vinujen Dilogen** | **Claude Code (Anthropic)** | System design and architecture planning; coding assistance, debugging, refactoring, testing, documentation, and ML/data/evaluation, RAG/grounding, and backend/frontend integration. AI-assisted code remains subject to review and technical defense. |
| **Vinujen Dilogen** | **ChatGPT (OpenAI)** | System design discussions, architecture and data-flow planning, comparison of implementation approaches, technical explanations, and interpretation of training/evaluation concepts. Also assisted with workflow and contribution documentation. |
| **Vinujen Dilogen** | **ChatGPT (OpenAI)** | Generated consumer-style queries and suggested recall IDs/relevance grades for **29 local draft annotations**. These are **AI-assisted drafts**, not independently human-written relevance judgments, and were not included in reported proxy metrics. |
| **Rukshan Baskaran** | **Claude Code (Anthropic)** | Team-reported use for system design, ML architecture, and planning the retrieval and training pipeline.  |
| **Rukshan Baskaran** | **ChatGPT (OpenAI)** | Team-reported use for system design discussions, comparison of ML approaches, and understanding retrieval and training workflows.  |
| **Ayngaraan Chandramohan** | **Claude Code (Anthropic)** | Team-reported use for system design and planning data preparation, preprocessing, and evaluation workflows.  |
| **Ayngaraan Chandramohan** | **ChatGPT (OpenAI)** | Team-reported use for system design discussions, data-flow planning, and exploration of preprocessing and evaluation approaches. |


The course guidelines permit AI assistance for brainstorming, debugging, code explanation, documentation, and comparison of approaches, but **not for producing report prose or experimental results**. Each member remains responsible for reviewing, understanding, reproducing, and modifying the submitted work. The written checkpoint report must be prepared by the team in accordance with this policy.
