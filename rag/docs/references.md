# RAG Benchmarks & Reference Systems

A curated list of benchmarks, state-of-the-art RAG systems, and key papers for reference.

---

## Benchmarks

### Domain-Specific QA Benchmarks

| Benchmark | Domain | Size | Task | Link |
|-----------|--------|------|------|------|
| **FinanceBench** | Finance | 150 Q | SEC filing QA | [Paper](https://arxiv.org/abs/2311.11944) |
| **PubMedQA** | Healthcare | 1000 Q | Biomedical yes/no | [Site](https://pubmedqa.github.io/) |
| **CUAD** | Legal | 13K Q | Contract clause extraction | [Site](https://www.atticusprojectai.org/cuad) |
| **BioASQ** | Biomedical | 4K Q | Literature QA | [Site](http://bioasq.org/) |
| **LegalBench** | Legal | 162 tasks | Legal reasoning | [Paper](https://arxiv.org/abs/2308.11462) |

### General RAG Benchmarks

| Benchmark | Focus | Size | Link |
|-----------|-------|------|------|
| **BEIR** | Retrieval across domains | 18 datasets | [Paper](https://arxiv.org/abs/2104.08663) |
| **KILT** | Knowledge-intensive tasks | 11 datasets | [Paper](https://arxiv.org/abs/2009.02252) |
| **MTEB** | Embedding evaluation | 56 datasets | [Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) |
| **RAGAS** | RAG evaluation metrics | - | [Docs](https://docs.ragas.io/) |
| **RGB** | RAG benchmark | 4 tasks | [Paper](https://arxiv.org/abs/2309.01431) |

### Multi-Hop & Complex QA

| Benchmark | Focus | Link |
|-----------|-------|------|
| **HotpotQA** | Multi-hop reasoning | [Paper](https://arxiv.org/abs/1809.09600) |
| **MuSiQue** | Multi-step reasoning | [Paper](https://arxiv.org/abs/2108.00573) |
| **StrategyQA** | Implicit reasoning | [Paper](https://arxiv.org/abs/2101.02235) |

---

## State-of-the-Art RAG Systems

### Production Systems

| System | Organization | Key Innovation |
|--------|--------------|----------------|
| **Perplexity** | Perplexity AI | Real-time web RAG |
| **Bing Chat** | Microsoft | Web search + GPT-4 |
| **ChatGPT Retrieval** | OpenAI | File upload RAG |
| **Claude Projects** | Anthropic | Document context |
| **NotebookLM** | Google | Multi-document synthesis |

### Open Source RAG Frameworks

| Framework | Best For | Link |
|-----------|----------|------|
| **LangChain** | Prototyping, chains | [GitHub](https://github.com/langchain-ai/langchain) |
| **LlamaIndex** | Data connectors | [GitHub](https://github.com/run-llama/llama_index) |
| **Haystack** | Production pipelines | [GitHub](https://github.com/deepset-ai/haystack) |
| **RAGatouille** | ColBERT retrieval | [GitHub](https://github.com/bclavie/RAGatouille) |
| **Chroma** | Vector DB | [GitHub](https://github.com/chroma-core/chroma) |
| **Weaviate** | Hybrid search | [GitHub](https://github.com/weaviate/weaviate) |

### Research Systems

| System | Innovation | Paper |
|--------|------------|-------|
| **REALM** | Pre-training with retrieval | [Paper](https://arxiv.org/abs/2002.08909) |
| **RAG (original)** | Retrieval-augmented generation | [Paper](https://arxiv.org/abs/2005.11401) |
| **RETRO** | Retrieval-enhanced transformers | [Paper](https://arxiv.org/abs/2112.04426) |
| **Atlas** | Few-shot with retrieval | [Paper](https://arxiv.org/abs/2208.03299) |
| **Self-RAG** | Self-reflective retrieval | [Paper](https://arxiv.org/abs/2310.11511) |

---

## Key Papers

### Foundational RAG

1. **RAG: Retrieval-Augmented Generation** (2020)
   - Lewis et al., Facebook AI
   - Original RAG formulation
   - [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)

2. **Dense Passage Retrieval (DPR)** (2020)
   - Karpukhin et al., Facebook AI
   - Bi-encoder dense retrieval
   - [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)

3. **ColBERT** (2020)
   - Khattab & Zaharia, Stanford
   - Late interaction for efficient retrieval
   - [arXiv:2004.12832](https://arxiv.org/abs/2004.12832)

### Advanced Retrieval

4. **BGE Embeddings** (2023)
   - BAAI
   - State-of-the-art embeddings
   - [arXiv:2309.07597](https://arxiv.org/abs/2309.07597)

5. **E5 Embeddings** (2022)
   - Microsoft
   - Text embeddings via contrastive learning
   - [arXiv:2212.03533](https://arxiv.org/abs/2212.03533)

6. **Hybrid Search** (2021)
   - Various
   - BM25 + Dense combination
   - [Survey](https://arxiv.org/abs/2112.01488)

### RAG Improvements

7. **Self-RAG** (2023)
   - Asai et al.
   - Self-reflective retrieval-augmented generation
   - [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)

8. **CRAG: Corrective RAG** (2024)
   - Yan et al.
   - Self-correcting retrieval
   - [arXiv:2401.15884](https://arxiv.org/abs/2401.15884)

9. **RAG-Fusion** (2023)
   - Multiple query generation + RRF
   - [Blog](https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1)

### Meta-Learning (Your Research Area)

10. **MAML** (2017)
    - Finn et al., Berkeley
    - Model-agnostic meta-learning
    - [arXiv:1703.03400](https://arxiv.org/abs/1703.03400)

11. **Toolformer** (2023)
    - Schick et al., Meta
    - LLMs learning to use tools
    - [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)

12. **Routing in MoE** (2022)
    - Various
    - Expert routing mechanisms
    - [Survey](https://arxiv.org/abs/2209.01667)

---

## Baseline Performance

### FinanceBench Baselines

| Method | Accuracy | Source |
|--------|----------|--------|
| GPT-4 (no RAG) | 19% | FinanceBench paper |
| GPT-4 + RAG | 51% | FinanceBench paper |
| Claude + RAG | ~55% | Community reports |
| **Our best** | **70.4%** | Element-based + forced answer |

### MTEB Embedding Leaderboard (Top 5)

| Model | Avg Score | Params |
|-------|-----------|--------|
| voyage-3 | 67.24 | - |
| text-embedding-3-large | 64.59 | - |
| bge-large-en-v1.5 | 64.23 | 335M |
| e5-large-v2 | 62.25 | 335M |
| gte-large | 63.13 | 335M |

*Note: We use `bge-large-en-v1.5` - solid choice, open source*

### Reranker Performance

| Model | BEIR nDCG@10 | Speed |
|-------|--------------|-------|
| bge-reranker-v2-m3 | 54.8 | Fast |
| bge-reranker-large | 53.6 | Medium |
| cross-encoder/ms-marco | 52.1 | Slow |

*Note: We use `bge-reranker-large` - good balance*

---

## Evaluation Metrics

### Retrieval Metrics

| Metric | What it measures |
|--------|------------------|
| **Recall@K** | % of relevant docs in top-K |
| **MRR** | Mean reciprocal rank of first relevant |
| **nDCG@K** | Normalized discounted cumulative gain |
| **Hit Rate** | Did we retrieve any relevant doc? |

### Generation Metrics

| Metric | What it measures |
|--------|------------------|
| **Exact Match (EM)** | Binary: exact string match |
| **F1 Score** | Token overlap with gold answer |
| **Semantic Similarity** | Embedding cosine similarity |
| **LLM Judge** | LLM rates answer quality |
| **ROUGE-L** | Longest common subsequence |
| **BERTScore** | Contextual embedding similarity |

### RAG-Specific Metrics (RAGAS)

| Metric | What it measures |
|--------|------------------|
| **Faithfulness** | Is answer grounded in context? |
| **Answer Relevancy** | Does answer address question? |
| **Context Precision** | Are retrieved docs relevant? |
| **Context Recall** | Did we get all needed info? |

---

## Quick Reference: Our System vs SOTA

| Component | Our Choice | Alternative | Why |
|-----------|------------|-------------|-----|
| **Embeddings** | BGE-large-en-v1.5 | E5, OpenAI | Open source, top-tier |
| **Reranker** | BGE-reranker-large | Cohere, ColBERT | Good accuracy/speed |
| **Vector DB** | ChromaDB | Pinecone, Weaviate | Simple, local |
| **LLM** | Llama 3.1 70B | GPT-4, Claude | Free on cluster |
| **Chunking** | Element-based | Fixed-size | Preserves structure |

---

## Useful Links

- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - Embedding rankings
- [Papers With Code - QA](https://paperswithcode.com/task/question-answering) - Latest QA papers
- [Awesome RAG](https://github.com/frutik/Awesome-RAG) - Curated RAG resources
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/) - Practical guide
- [LlamaIndex RAG](https://docs.llamaindex.ai/en/stable/understanding/rag/) - Alternative framework
