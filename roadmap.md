## RAG + Meta-Learning Roadmap

### 1) Upgrade Retrieval for Finance
- Swap reranker to a stronger model (e.g., `bge-reranker-large` or Cohere Rerank) with caching.
- Improve chunk metadata: include company/year/doc_type in metadata (not just filename); expand company/year detection beyond the hardcoded list; ingest FinanceBench `financebench_document_information.jsonl` to tag chunks.
- Add higher-k/hierarchical option: doc/section prefilter before chunk retrieval; expose `initial_k_factor` per pipeline.
- Align FinanceBench QA: use `financebench_open_source.jsonl` instead of the 24-question subset; measure EM/F1 for numeric answers.
- Status: PDFs + full QA + doc metadata downloaded. Adapter and index not yet rebuilt to use the full JSONL/metadata; current runs ~0.5 semantic similarity on 150 Q with old setup.

### 2) Evidence & Generation Discipline
- Add “must-cite spans” prompt variant; keep both variants and compare.
- Add a lightweight numeric consistency check: if answer is numeric, scan retrieved text for nearby numbers/units and flag mismatches.
- Tighten numeric formatting for FinanceBench (answer-only with units for EM).

### 3) Meta-Learning Plumbing
- Implement `meta_learning/oracle_labels.py`: grid over pipelines on training splits, emit best pipeline per question.
- Implement router + trainer (`meta_learning/router.py`, `meta_learning/meta_trainer.py`): input = question (+ optional support stats), output = `pipeline_id`.
- Add episodic evaluator: sample tasks with support/query splits; report `Ans`, `Recall@k`, `ToolAcc`; baselines = fixed pipelines, per-domain tuned, random, oracle.

### 4) New Domains (Beyond Finance)
- Target domains: FinanceBench (existing), PubMedQA (biomedical), CUAD (legal), ScienceQA (text-only slice). If keeping to three, drop ScienceQA.
- Add adapters/indexes for PubMedQA and CUAD; optionally ScienceQA. Wire into `corpora/index_manager.py`.
- Verify pipelines run end-to-end on each domain (semantic/hybrid/filter/rerank).

### 5) Benchmarks & Experiments
- Domains: FinanceBench + PubMedQA + CUAD + ScienceQA (text slice). If needed, drop ScienceQA to keep scope at three.
- Metrics: `Ans` (semantic similarity/EM), `Recall@k`, `ToolAcc`.
- Baselines: always-semantic, always-hybrid, always-full (`hybrid_filter_rerank`), best fixed per domain, random pipeline, oracle best-per-question.
- Ours: meta-router (support-agnostic and support-aware).
- Plan after all domains are indexed:
  - Generate oracle tool labels per domain via pipeline grid search.
  - Train router on labels (question-only, then support-aware).
  - Episodic eval: within-domain and cross-domain; report `Ans`, `Recall@5`, `ToolAcc`.
  - Hold-out eval: train on three domains, test on held-out domain for zero/few-shot adaptation.
  - Ablations: support size, tool availability (filter/rerank), reranker choice, `initial_k_factor`.

### 6) Docs & Tests
- Update README for pipeline usage and new domain.
- Add smoke tests: `test_pipelines.py` over both domains with tiny subsets.

### References / Links
- **Finance/General RAG**: FiD https://arxiv.org/abs/2007.01282; Atlas https://arxiv.org/abs/2208.03299; REALM https://arxiv.org/abs/2002.08909; DPR https://arxiv.org/abs/2004.04906; BEIR https://arxiv.org/abs/2104.08663
- **Tool use**: Toolformer https://arxiv.org/abs/2302.04761; ReAct https://arxiv.org/abs/2210.03629
- **Retrievers/Rerankers**: BGE https://arxiv.org/abs/2309.07597 (models: https://huggingface.co/BAAI)
- **Meta-learning**: MAML https://arxiv.org/abs/1703.03400; Prototypical Networks https://arxiv.org/abs/1703.05175; MetaICL https://arxiv.org/abs/2205.12755
- **Benchmarks**:
  - FinanceBench (existing in repo)
  - PubMedQA: https://pubmedqa.github.io/ ; HF: https://huggingface.co/datasets/pubmedqa
  - CUAD: https://www.atticusprojectai.org/cuad ; HF: https://huggingface.co/datasets/theatticusproject/cuad
  - ScienceQA: https://scienceqa.github.io/ ; HF: https://huggingface.co/datasets/derek-thomas/ScienceQA
