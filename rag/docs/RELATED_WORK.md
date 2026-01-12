# Related Work: RAG Routing & Meta-Learning

> Papers and GitHub repos directly applicable to your FinanceBench RAG routing research.
> Downloaded: 2026-01-09
> Updated with ICLR 2025 papers and top GitHub repos

---

## 1. Adaptive-RAG (KAIST, 2024) - **MOST RELEVANT**

**Title:** Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity

**Authors:** Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, Jong C. Park

**arXiv:** https://arxiv.org/abs/2403.14403

### Key Idea
Train a **lightweight classifier** to predict query complexity, then route to appropriate strategy:
- **Simple queries** → No retrieval (LLM parametric knowledge)
- **Medium queries** → Single-step retrieval
- **Complex queries** → Iterative multi-step retrieval

### Why It's Relevant to You
- They classify by **complexity**; you classify by **question type** (metrics/domain/novel)
- Their classifier uses auto-collected labels from model outcomes
- You could generate oracle labels the same way: run all pipelines, pick best

### Method Details
```
Query → Complexity Classifier (small LM) → Route Decision
                                              ├─ No retrieval
                                              ├─ Single-step RAG
                                              └─ Iterative RAG
```

### Key Quote
> "We train a smaller language model classifier using automatically collected labels derived from actual predicted outcomes of models and inherent inductive biases in datasets."

---

## 2. RouteRAG (CAS, Dec 2025) - **RL-BASED ROUTING**

**Title:** RouteRAG: Efficient Retrieval-Augmented Generation from Text and Graph via Reinforcement Learning

**Authors:** Yucan Guo, Miao Su, Saiping Guan, Zihao Sun, Xiaolong Jin, Jiafeng Guo, Xueqi Cheng

**arXiv:** https://arxiv.org/abs/2512.09487

### Key Idea
Use **reinforcement learning** to learn when to:
- Reason (no retrieval)
- Retrieve from text corpus
- Retrieve from knowledge graph
- Produce final answer

### Why It's Relevant to You
- They route between **text vs graph**; you route between **pipelines**
- Two-stage training: (1) task outcome, (2) retrieval efficiency
- Balances quality with cost (fewer retrievals = faster)

### Method Details
```
Query → RL Policy Agent → Action Selection
                            ├─ Reason (think step)
                            ├─ Retrieve(text)
                            ├─ Retrieve(graph)
                            └─ Answer
```

### Key Quote
> "RouteRAG jointly optimizes the entire generation process via RL, allowing the model to determine when to reason, what to retrieve from either texts or graphs, and when to produce final answers."

---

## 3. Self-RAG (UW + IBM, 2023) - **ADAPTIVE RETRIEVAL**

**Title:** Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection

**Authors:** Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi

**arXiv:** https://arxiv.org/abs/2310.11511

### Key Idea
Train LLM to emit **reflection tokens** that control retrieval:
- `[Retrieve]` - Should I retrieve?
- `[IsRel]` - Is retrieved doc relevant?
- `[IsSup]` - Does retrieved doc support answer?
- `[IsUse]` - Is output useful?

### Why It's Relevant to You
- On-demand retrieval (not always retrieve)
- Self-critique improves factuality
- 7B model outperforms ChatGPT on QA tasks

### Method Details
```
Query → LLM generates [Retrieve] token
         ├─ Yes → Retrieve → Generate with [IsSup] critique
         └─ No  → Generate directly
```

### Key Quote
> "Self-RAG enhances an LM's quality and factuality through retrieval and self-reflection, using special reflection tokens that make the LM controllable during inference."

---

## 4. CRAG - Corrective RAG (USTC, 2024) - **RETRIEVAL QUALITY**

**Title:** Corrective Retrieval Augmented Generation

**Authors:** Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling

**arXiv:** https://arxiv.org/abs/2401.15884

### Key Idea
Add a **retrieval evaluator** that assesses document quality:
- **Correct** → Use retrieved docs
- **Ambiguous** → Combine with web search
- **Incorrect** → Trigger web search fallback

### Why It's Relevant to You
- Evaluator can detect when retrieval fails
- Could detect when table chunks are fragmented
- Plug-and-play enhancement for existing RAG

### Method Details
```
Query → Retrieve → Evaluator → Confidence Score
                                 ├─ High   → Use docs
                                 ├─ Medium → docs + web search
                                 └─ Low    → Web search only
```

### Key Quote
> "CRAG includes a lightweight retrieval evaluator that assesses document quality and triggers different retrieval actions based on confidence levels."

---

## 5. Meta-Learning Survey (2023) - **BACKGROUND**

**Title:** Meta-learning approaches for few-shot learning: A survey of recent advances

**Authors:** Hassan Gharoun, Fereshteh Momenifar, Fang Chen, Amir H. Gandomi

**arXiv:** https://arxiv.org/abs/2303.07502

### Three Categories of Meta-Learning

| Category | Methods | Your Application |
|----------|---------|------------------|
| **Metric-based** | ProtoNet, Matching Networks | Learn embedding space for question similarity |
| **Memory-based** | MANN, Neural Turing Machines | Store successful pipeline selections |
| **Learning-based** | MAML, Reptile | Learn initialization for fast adaptation |

### Key Insight for Your Work
> Recent research shows that with strong foundation model features, **simpler methods like ProtoNet can outperform MAML**. This validates your sklearn-based router approach!

---

## 6. RAG Survey (Tongji, 2023) - **TAXONOMY**

**Title:** Retrieval-Augmented Generation for Large Language Models: A Survey

**Authors:** Yunfan Gao et al.

**arXiv:** https://arxiv.org/abs/2312.10997

### RAG Paradigm Taxonomy

```
RAG Evolution:
├─ Naive RAG      → Simple retrieve-then-read
├─ Advanced RAG   → Pre/post-retrieval optimization
└─ Modular RAG    → Flexible component composition (YOUR APPROACH)
```

### Modular RAG Components
1. **Retrieval** - How to find relevant docs
2. **Augmentation** - How to enhance with retrieved info
3. **Generation** - How to produce final answer

Your system is **Modular RAG** with learned routing between modules.

---

## How These Papers Map to Your Work

| Paper | Their Focus | Your Adaptation |
|-------|-------------|-----------------|
| Adaptive-RAG | Query complexity routing | Question TYPE routing (metrics/domain/novel) |
| RouteRAG | RL for text/graph routing | RL for pipeline routing |
| Self-RAG | Reflection tokens | Could add retrieval quality tokens |
| CRAG | Retrieval correction | Detect table fragmentation |
| Meta-Learning Survey | Few-shot methods | Train router on oracle labels |

---

## Your Novel Contribution

What existing work **doesn't** address:

1. **Table-aware routing** - No one routes based on whether query needs table data
2. **Question semantics** - Others route by complexity, not by question TYPE
3. **Empirical model ceiling** - No one has shown all models hit same ceiling
4. **Chunking as root cause** - No systematic study of chunking → retrieval quality

### Your Paper Positioning
```
EXISTING:  Query Complexity → Route to Retrieval Strategy
YOURS:     Question Type + Table Detection → Route to Optimal Pipeline
           + Empirical proof that model selection doesn't matter
           + Fix: Table-aware chunking > model scaling
```

---

## Recommended Reading Order

1. **Adaptive-RAG** (1 hour) - Most similar to your approach
2. **Self-RAG** (1 hour) - Elegant reflection token idea
3. **CRAG** (30 min) - Simple but effective retrieval correction
4. **RouteRAG** (1 hour) - If you want RL-based routing
5. **Meta-Learning Survey** (skim) - Background on MAML vs simpler methods

---

## BibTeX for Your Paper

```bibtex
@article{jeong2024adaptive,
  title={Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity},
  author={Jeong, Soyeong and Baek, Jinheon and Cho, Sukmin and Hwang, Sung Ju and Park, Jong C},
  journal={arXiv preprint arXiv:2403.14403},
  year={2024}
}

@article{guo2025routerag,
  title={RouteRAG: Efficient Retrieval-Augmented Generation from Text and Graph via Reinforcement Learning},
  author={Guo, Yucan and Su, Miao and Guan, Saiping and others},
  journal={arXiv preprint arXiv:2512.09487},
  year={2025}
}

@article{asai2023self,
  title={Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection},
  author={Asai, Akari and Wu, Zeqiu and Wang, Yizhong and Sil, Avirup and Hajishirzi, Hannaneh},
  journal={arXiv preprint arXiv:2310.11511},
  year={2023}
}

@article{yan2024corrective,
  title={Corrective Retrieval Augmented Generation},
  author={Yan, Shi-Qi and Gu, Jia-Chen and Zhu, Yun and Ling, Zhen-Hua},
  journal={arXiv preprint arXiv:2401.15884},
  year={2024}
}

@article{gharoun2023meta,
  title={Meta-learning approaches for few-shot learning: A survey of recent advances},
  author={Gharoun, Hassan and Momenifar, Fereshteh and Chen, Fang and Gandomi, Amir H},
  journal={arXiv preprint arXiv:2303.07502},
  year={2023}
}

@article{gao2023retrieval,
  title={Retrieval-Augmented Generation for Large Language Models: A Survey},
  author={Gao, Yunfan and Xiong, Yun and others},
  journal={arXiv preprint arXiv:2312.10997},
  year={2023}
}
```

---

# ICLR 2025 Papers (Newly Added)

## 7. Speculative RAG (ICLR 2025 Main) - **EFFICIENCY**

**Title:** Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting

**Authors:** Zilong Wang, Zifeng Wang, Long Le, et al. (Google)

**arXiv:** https://arxiv.org/abs/2407.08223

### Key Idea
Use a **smaller specialist LM** to draft multiple responses in parallel, then a **larger generalist LM** to verify. Each draft uses a different subset of retrieved documents.

### Why It's Relevant
- +12.97% accuracy, -50.83% latency vs conventional RAG
- Parallel drafting = diverse perspectives on evidence
- Could apply to your pipeline: draft with each pipeline, verify best

### Method
```
Query → Retrieve Docs → Split into Subsets
                            ↓
         ┌──────────────────┼──────────────────┐
         ↓                  ↓                  ↓
    Draft 1 (specialist)  Draft 2           Draft 3
         └──────────────────┼──────────────────┘
                            ↓
                    Verify (generalist LM)
                            ↓
                       Best Answer
```

---

## 8. Optimizing RAG for Finance (ICLR 2025 FinAI Workshop) - **DIRECT COMPETITOR**

**Title:** Optimizing Retrieval Strategies for Financial Question Answering Documents in RAG Systems

**Authors:** Sejong Kim, Hyunseo Song, Hyunwoo Seo, Hyunjun Kim

**arXiv:** https://arxiv.org/abs/2503.15191

### Key Idea
Three-phase pipeline for financial RAG:
1. **Pre-retrieval:** Query expansion + corpus markdown restructuring
2. **Retrieval:** Fine-tuned embeddings + hybrid dense/sparse
3. **Post-retrieval:** DPO training + document selection

### Why It's Relevant
- Tested on **FinanceBench** (same as you!)
- Addresses "multi-hierarchical tabular data" challenges
- Uses hybrid retrieval (you do too)

### Your Differentiation
| Their Work | Your Work |
|------------|-----------|
| Query expansion | Question-type routing |
| Fine-tuned embeddings | Table-aware chunking |
| DPO post-training | No fine-tuning needed |
| No model comparison | Prove model doesn't matter |

---

## 9. FinanceBench (Original Benchmark) - **YOUR DATASET**

**Title:** FinanceBench: A New Benchmark for Financial Question Answering

**Authors:** Pranab Islam, Anand Kannappan, Douwe Kiela, Rebecca Qian, Nino Scherrer, Bertie Vidgen

**arXiv:** https://arxiv.org/abs/2311.11944

### Key Stats
- 10,231 questions about publicly traded companies
- GPT-4-Turbo + retrieval: **81% incorrect or refused**
- All models exhibit hallucinations

### Question Types (150 sample)
- **Metrics-generated:** Extract specific numbers from tables
- **Domain-relevant:** Require domain knowledge
- **Novel-generated:** Require reasoning/synthesis

---

## 10. Metadata-Driven RAG (Oct 2025) - **CONTEXTUAL CHUNKS**

**Title:** Metadata-Driven Retrieval-Augmented Generation for Financial Question Answering

**Authors:** Michail Dadopoulos, Anestis Ladas, Stratos Moschidis, Ioannis Negkakis

**arXiv:** https://arxiv.org/abs/2510.24402

### Key Finding
> "The most significant performance gains come from **embedding chunk metadata directly with text**"

### What Are Contextual Chunks?
Instead of just embedding text, embed text + metadata together:
```
Traditional: "Revenue was $383B" → embed
Contextual:  "Apple Inc. | FY2023 | 10-K | Revenue was $383B" → embed
```

### Why It's Relevant
- Tested on FinanceBench
- Shows **chunking strategy > reranker quality**
- Validates your focus on chunking over model selection

---

## 11. Financial Report Chunking (Feb 2024) - **STRUCTURE-AWARE**

**Title:** Financial Report Chunking for Effective Retrieval Augmented Generation

**Authors:** Antonio Jimeno Yepes, Yao You, Jan Milczek, Sebastian Laverde, Renyu Li

**arXiv:** https://arxiv.org/abs/2402.05131

### Key Idea
**Chunk by structural element type**, not by paragraph/characters:
- Tables → keep as atomic units
- Sections → respect boundaries
- Headers → attach to content

### Why It's Critical for You
This paper validates your exact approach:
> "Dissecting documents into constituent elements creates a way to chunk that yields optimal chunk size **without tuning**."

Your `ingest.py` already does this! (Tables kept intact, prose chunked separately)

---

# Top GitHub Repositories

## 12. dsRAG - **96.6% on FinanceBench!**

**Repo:** https://github.com/D-Star-AI/dsRAG

### Why It's Relevant
- Achieves **96.6%** on FinanceBench (vs baseline 19-32%)
- Uses Claude 3.5 Sonnet
- High-performance retrieval engine for unstructured data

### What to Learn
- Their chunking strategy
- How they handle tables
- Retrieval pipeline architecture

---

## 13. FinanceRAG (ACM-ICAIF '24 Winner)

**Repo:** https://github.com/cv-lee/FinanceRAG

### Why It's Relevant
- Competition-winning system
- Ablation studies on query expansion
- Multiple reranker ensemble

### Key Techniques
- Pre-retrieval: Query expansion, corpus refinement
- Retrieval: Multiple reranker models
- Generation: Long context management

---

## 14. FinSage - **92.51% Recall on FinanceBench**

**Repo:** https://github.com/HFHL/finsage

### Why It's Relevant
- Addresses data heterogeneity (text, tables, charts)
- +24.06% over best baseline
- Intelligent framework for financial compliance

---

## 15. RAG-Multimodal-Financial-Document-Analysis

**Repo:** https://github.com/Mattral/RAG-Multimodal-Financial-Document-Analysis-and-Recall

### Why It's Relevant
- Uses Unstructured.io for text/table extraction (same as you!)
- GPT-4V for chart understanding
- LlamaIndex for retrieval

---

## 16. Official FinanceBench Dataset

**Repo:** https://github.com/patronus-ai/financebench

### Why It's Essential
- Official benchmark implementation
- 150 test cases available
- Evaluation scripts

---

## 17. IBM RAG Chunking Techniques

**Repo:** https://github.com/IBM/rag-chunking-techniques

### Why It's Relevant
- Systematic evaluation of chunking strategies
- Company policies data (similar to financial docs)
- Smart chunking implementation

---

## 18. Jina Late Chunking

**Repo:** https://github.com/jina-ai/late-chunking

### Why It's Relevant
- Novel chunking approach
- Maintains context across chunk boundaries
- Could improve your table handling

---

## 19. Chroma Chunking Evaluation - **BENCHMARK STUDY**

**URL:** https://research.trychroma.com/evaluating-chunking

### What They Tested
| Strategy | Type | Description |
|----------|------|-------------|
| RecursiveCharacterTextSplitter | Simple | Standard LangChain splitter |
| TokenTextSplitter | Simple | Token-based splitting |
| KamradtSemanticChunker | Semantic | Greg Kamradt's approach |
| KamradtModifiedChunker | Semantic | + Binary search for target size |
| ClusterSemanticChunker | Semantic | Maximize similarity within chunks |
| LLMSemanticChunker | LLM-based | LLM identifies split points |

### Key Results (text-embedding-3-large)
| Strategy | Recall | Notes |
|----------|--------|-------|
| **LLMSemanticChunker** | **91.9%** | Highest recall, expensive |
| **ClusterSemanticChunker** (200 tokens) | 91.3% | Best precision/efficiency |
| RecursiveCharacterTextSplitter (200 tokens) | ~89% | Solid baseline |

### Critical Finding
> "The choice of chunking strategy can have a significant impact on retrieval performance, with some strategies outperforming others by **up to 9% in recall**."

### Recommendation
- **For most cases:** RecursiveCharacterTextSplitter at 200-400 tokens works well
- **For precision:** ClusterSemanticChunker with smaller chunks (200 tokens)
- **Key insight:** Chunk size matters more than the algorithm!

### Why This Matters for You
- Validates that **chunking strategy = 9% recall difference**
- Your table-aware chunking is a **structural** approach (not tested here!)
- This study focuses on prose - **your table insight is novel**

---

# Summary: Priority Reading List

## Must Read (This Week)
| Priority | Paper/Repo | Time | Why |
|----------|-----------|------|-----|
| 1 | [Adaptive-RAG](https://arxiv.org/abs/2403.14403) | 1h | Query routing baseline |
| 2 | [Financial Report Chunking](https://arxiv.org/abs/2402.05131) | 30m | Validates your approach |
| 3 | [dsRAG repo](https://github.com/D-Star-AI/dsRAG) | 1h | 96.6% FinanceBench |
| 4 | [Metadata-Driven RAG](https://arxiv.org/abs/2510.24402) | 30m | Contextual chunks |

## Should Read (Next Week)
| Priority | Paper/Repo | Time | Why |
|----------|-----------|------|-----|
| 5 | [ICLR FinAI RAG](https://arxiv.org/abs/2503.15191) | 1h | Direct competitor |
| 6 | [Speculative RAG](https://arxiv.org/abs/2407.08223) | 30m | Efficiency gains |
| 7 | [FinanceRAG repo](https://github.com/cv-lee/FinanceRAG) | 1h | Competition winner |
| 8 | [Self-RAG](https://arxiv.org/abs/2310.11511) | 1h | Reflection tokens |

---

# New BibTeX Entries

```bibtex
@inproceedings{wang2025speculative,
  title={Speculative RAG: Enhancing Retrieval Augmented Generation through Drafting},
  author={Wang, Zilong and Wang, Zifeng and Le, Long and others},
  booktitle={ICLR},
  year={2025}
}

@article{kim2025optimizing,
  title={Optimizing Retrieval Strategies for Financial Question Answering Documents in RAG Systems},
  author={Kim, Sejong and Song, Hyunseo and Seo, Hyunwoo and Kim, Hyunjun},
  journal={ICLR 2025 Workshop on Financial AI},
  year={2025}
}

@article{islam2023financebench,
  title={FinanceBench: A New Benchmark for Financial Question Answering},
  author={Islam, Pranab and Kannappan, Anand and Kiela, Douwe and others},
  journal={arXiv preprint arXiv:2311.11944},
  year={2023}
}

@article{dadopoulos2025metadata,
  title={Metadata-Driven Retrieval-Augmented Generation for Financial Question Answering},
  author={Dadopoulos, Michail and Ladas, Anestis and Moschidis, Stratos and Negkakis, Ioannis},
  journal={arXiv preprint arXiv:2510.24402},
  year={2025}
}

@article{yepes2024financial,
  title={Financial Report Chunking for Effective Retrieval Augmented Generation},
  author={Yepes, Antonio Jimeno and You, Yao and Milczek, Jan and Laverde, Sebastian and Li, Renyu},
  journal={arXiv preprint arXiv:2402.05131},
  year={2024}
}
```
