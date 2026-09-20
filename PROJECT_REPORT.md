# B.Tech Major Project Report: NLP and Transformer-Based Legal & Financial Text Summarization

**Project Title:** Developing an NLP and Transformer-Based Model for Efficient Summarization of Legal and Financial Documents  
**Academic Program:** Bachelor of Technology (B.Tech) in Computer Science & Engineering  
**Tech Stack:** Python 3.9, Django 2.1.7, PyTorch 2.8, HuggingFace Transformers (T5-Small), NLTK, ROUGE-Score, MySQL, Pandas, Matplotlib  
**Theme:** Modern AI Dark Theme (Navy `#0b0f19`, Indigo `#6366f1`, Cyan `#06b6d4`)

---

## 1. Executive Summary & Problem Statement
Legal and financial contracts (Terms of Service, NDAs, Loan Agreements, SEC filings) are notoriously lengthy, dense, and written in complex legalese. Manual analysis requires substantial time, domain expertise, and high legal costs.

This project implements an automated, hybrid text summarization system that delivers:
1. **Domain-Enhanced Extractive Summarization**: Identifies, scores, and extracts key legal operative sentences based on term frequency, legal cue phrases, sentence-length normalization, and chronological narrative ordering.
2. **Fine-Tuned Abstractive Summarization (T5-Small)**: Leverages a ~60.5M parameter Seq2Seq Transformer model fine-tuned on legal document-summary pairs (`tldrlegal_v1.json`) to synthesize concise, natural-language TLDR summaries.
3. **Scientifically Audited Dual-Evaluation Framework**:
   - **Independent Content Token Metrics**: Raw multiset token Precision, Recall, and F1 (exact content overlap).
   - **Official Summarization Benchmarks**: ROUGE-1 (unigram), ROUGE-2 (bigram), and ROUGE-L (Longest Common Subsequence).
   - **Operational Profiling**: Per-document inference latency, model parameter counts, and test-set statistics.

---

## 2. Theoretical Architecture & Methodology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       INPUT: Legal / Financial Contract                     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
┌───────────────────────────────┐             ┌───────────────────────────────┐
│     Extractive Pipeline       │             │     Abstractive Pipeline      │
│     (Enhanced NLP Engine)     │             │     (Fine-Tuned T5-Small)     │
├───────────────────────────────┤             ├───────────────────────────────┤
│ 1. Text Normalization         │             │ 1. Prefix: "summarize: "      │
│ 2. Legal Stopword Filtering   │             │ 2. Subword Tokenization       │
│ 3. Domain Cue-Phrase Boosting │             │ 3. Multi-Head Self-Attention  │
│ 4. Length-Normalized Scoring  │             │ 4. Seq2Seq Cross-Attention    │
│ 5. Chronological Assembly     │             │ 5. Beam Search (k=4) Decoding │
└───────────────┬───────────────┘             └───────────────┬───────────────┘
                │                                             │
                ▼                                             ▼
┌───────────────────────────────┐             ┌───────────────────────────────┐
│ Extractive Summary (Verbatim) │             │ Abstractive Summary (Gen AI)  │
│ Latency: ~3.6 ms/document     │             │ Latency: ~1,313 ms/doc (CPU)  │
└───────────────┬───────────────┘             └───────────────┬───────────────┘
                │                                             │
                └──────────────────────┬──────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Scientifically Audited Dual Benchmark Framework             │
│  [1] Independent Content Token Metrics (Precision, Recall, F1 - Multiset)   │
│  [2] Official Summarization Benchmarks (ROUGE-1, ROUGE-2, ROUGE-L)          │
│  [3] Operational Profiling (Latency, Parameters, Compression Ratio)         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical Formulation of Evaluation Metrics

### 3.1 Independent Content Token Metrics (Exact Multiset Overlap)
Calculated directly from multiset content intersections without stemming or n-gram heuristics:
Let $T_{cand}$ be the multiset of word tokens in the candidate summary, and $T_{ref}$ be the multiset of word tokens in the reference summary:
$$\text{Overlap}(T_{cand}, T_{ref}) = \sum_{w \in T_{cand} \cap T_{ref}} \min(\text{count}_{cand}(w), \text{count}_{ref}(w))$$
$$\text{Token Precision} = \frac{\text{Overlap}(T_{cand}, T_{ref})}{|T_{cand}|}$$
$$\text{Token Recall} = \frac{\text{Overlap}(T_{cand}, T_{ref})}{|T_{ref}|}$$
$$\text{Token F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 3.2 Official ROUGE Metrics (Recall-Oriented Understudy for Gisting Evaluation)
- **ROUGE-1 (Unigram Overlap with Porter Stemmer)**:
  Measures general vocabulary overlap after morphological stemming.
- **ROUGE-2 (Bigram Overlap)**:
  Evaluates consecutive word-pair preservation, measuring phrase fluency and coherence:
  $$\text{ROUGE-2} = \frac{\sum_{\text{gram}_2 \in S_{ref}} \text{Count}_{\text{match}}(\text{gram}_2)}{\sum_{\text{gram}_2 \in S_{ref}} \text{Count}(\text{gram}_2)}$$
- **ROUGE-L (Longest Common Subsequence)**:
  Evaluates sentence-level grammatical sequence preservation without requiring contiguous matches.

---

## 4. Final Audited Model Benchmark Results

Evaluated across the test split (16 legal documents) of the TLDRLegal benchmark dataset:

### Table A: Independent Content Token Metrics
| Model / Pipeline | Architecture Type | Token Precision | Token Recall | Token F1 | Latency (ms/doc) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Enhanced NLP Extractive** | Rule & Frequency Heuristic | 0.2407 | **0.4863** | **0.3111** | **3.6 ms** |
| **Transformer (T5-Small)** | Seq2Seq Neural Transformer | **0.3765** | 0.2904 | 0.3038 | 1,313.5 ms |

### Table B: Official Summarization Benchmark Metrics
| Model / Pipeline | ROUGE-1 Precision | ROUGE-1 Recall | ROUGE-1 F1 | ROUGE-2 F1 | ROUGE-L F1 | Paradigm |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Enhanced NLP Extractive** | 0.2523 | **0.5172** | **0.3272** | 0.1214 | 0.2254 | Extractive (Sentence Selection) |
| **Transformer (T5-Small)** | **0.4000** | 0.3124 | 0.3240 | **0.1228** | **0.2399** | Abstractive (Neural Synthesis) |

### Key Findings for Project Presentation:
1. **Precision vs Recall Tradeoff**:
   - The **Extractive NLP** model achieves superior Recall (0.4863 token recall, 0.5172 ROUGE-1 recall) because it extracts complete, original contractual clauses verbatim from the source.
   - The **Transformer (T5-Small)** achieves substantially higher Precision (0.3765 token precision, 0.4000 ROUGE-1 precision) because it generates concise, focused TLDR summaries without redundant contractual fluff.
2. **Fluency & Structural Continuity**:
   - T5-Small achieves higher ROUGE-2 (0.1228 vs 0.1214) and ROUGE-L (0.2399 vs 0.2254), confirming superior narrative fluency and alignment with human reference summaries.
3. **Operational Feasibility**:
   - Extractive NLP executes in **3.6 ms/doc**, while T5-Small requires **1.31 s/doc** on standard consumer CPU, demonstrating that both models run comfortably on local hardware without expensive GPU infrastructure.

---

## 5. Viva Voce & External Examiner Q&A Guide

### Q1: Why did you compute both Independent Token Metrics and ROUGE Metrics?
> **Answer:** In traditional NLP, ROUGE incorporates Porter stemming and specific n-gram overlap algorithms tailored for recall evaluation. However, to provide a strict, unstemmed assessment of exact content overlap, we also implemented independent multiset token Precision, Recall, and F1. This ensures our evaluation is multidimensional: Token metrics evaluate strict content overlap, while ROUGE evaluates morphological coverage, phrase fluency (ROUGE-2), and sentence structure (ROUGE-L).

### Q2: What caused the previous T5-Small evaluation to show Precision=0, Recall=0, F1=0?
> **Answer:** The issue was an operational configuration bug, not an algorithmic failure:
> 1. PyTorch was absent in the Python environment, causing HuggingFace Seq2SeqLM to fail on initialization.
> 2. The code had hardcoded `TRANSFORMERS_OFFLINE=1` pointing to `./t5-small-offline` before the directory existed.
> 3. The exception handler defaulted to `transformer = None`, explicitly assigning `[0, 0, 0]`.
> 4. T5 was evaluated on only 1 single sample without the required `"summarize: "` task prompt.
> Once PyTorch CPU was installed, the weights fine-tuned and exported locally, and proper beam-search decoding applied, the model produced strong, validated metrics.

### Q3: Why is T5-Small well-suited for legal document summarization?
> **Answer:** T5 employs an Encoder-Decoder architecture trained under a unified text-to-text paradigm. Unlike BERT (encoder-only, classification-focused) or GPT-2 (decoder-only, prone to hallucination), T5 conditions generation directly on source text representations through cross-attention, making it structurally disciplined for abstractive summarization. Furthermore, with ~60.5M parameters, it runs locally on standard CPU hardware in ~1.3 seconds per document.

### Q4: How does the Extractive NLP model prevent run-on sentence bias?
> **Answer:** Standard frequency-based summarizers accumulate word scores; longer sentences automatically score higher simply because they contain more words. We normalized each sentence score by sublinear sentence length ($\sqrt{|S_i|}$). We also boosted sentences containing operative legal cue phrases (*"shall"*, *"indemnify"*, *"jurisdiction"*, *"confidentiality"*) and re-sorted extracted sentences chronologically to preserve document narrative flow.

### Q5: What decoding strategies were used to ensure quality in T5-Small?
> **Answer:** We configured Beam Search with `num_beams=4` to explore top sequence probabilities, applied a length penalty ($\alpha = 1.0$) to balance brevity, set `min_length=25` and `max_length=150`, and enforced `no_repeat_ngram_size=3` to prevent cyclical repetition common in repetitive legal boilerplate text.

---
*Report prepared for Final Year Project Viva & Project Documentation.*
