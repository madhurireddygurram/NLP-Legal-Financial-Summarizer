"""
Audited Dual-Model Benchmark & Profiling Script.
Computes:
1. Independent Content Token Metrics: Precision, Recall, F1 (Strictly F1 = 2 * P * R / (P + R))
2. Official Summarization Benchmarks: ROUGE-1, ROUGE-2, ROUGE-L (with Porter Stemmer)
3. Operational Metrics: Latency (ms/doc), Parameters, Architecture Type, Test Sample Count
Saves comprehensive audited results to eval_cache.json.
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import os
import json
import re
import math
import time
from collections import Counter
from heapq import nlargest
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.corpus import stopwords
from nltk import tokenize
from rouge_score import rouge_scorer

try:
    stop_words = set(stopwords.words('english'))
except Exception:
    import nltk
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

LEGAL_KEYWORDS = {
    'shall': 2.0, 'agree': 1.5, 'liable': 2.0, 'liability': 2.0, 'indemnify': 2.0,
    'indemnification': 2.0, 'warranty': 1.8, 'warranties': 1.8, 'jurisdiction': 1.8,
    'governed': 1.8, 'governing': 1.8, 'termination': 2.0, 'terminate': 1.8,
    'confidential': 2.0, 'confidentiality': 2.0, 'breach': 2.0, 'intellectual': 1.8,
    'property': 1.5, 'rights': 1.8, 'obligation': 1.8, 'obligations': 1.8,
    'dispute': 1.8, 'arbitration': 2.0, 'damages': 1.8, 'payment': 1.8,
    'effective': 1.5, 'compliance': 1.8, 'laws': 1.5, 'provision': 1.5
}


def clean_legal_text(text):
    if not text:
        return ""
    text = text.replace('\xa0', ' ').replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
    return re.sub(r'\s+', ' ', text).strip()


def calculate_token_metrics(reference, candidate):
    """
    Independent Token-Level Precision and Recall.
    Calculates exact multiset content overlap independent from ROUGE stemmer/n-gram heuristics.
    """
    ref_tokens = re.findall(r'\b[a-zA-Z0-9_]+\b', reference.lower())
    cand_tokens = re.findall(r'\b[a-zA-Z0-9_]+\b', candidate.lower())
    if not cand_tokens or not ref_tokens:
        return 0.0, 0.0
    ref_counts = Counter(ref_tokens)
    cand_counts = Counter(cand_tokens)
    overlap = sum((cand_counts & ref_counts).values())
    p = overlap / len(cand_tokens)
    r = overlap / len(ref_tokens)
    return p, r


def harmonic_f1(p, r):
    """Ensures exact F1 = 2 * P * R / (P + R) consistency."""
    if (p + r) > 0:
        return round((2 * p * r) / (p + r), 4)
    return 0.0


def summarize_extractive(essay, threshold=0.35):
    essay = clean_legal_text(essay)
    if not essay:
        return ""
    sentence_tokens = tokenize.sent_tokenize(essay)
    if len(sentence_tokens) <= 2:
        return essay

    words = re.findall(r'[a-zA-Z]{2,}', essay.lower())
    word_frequencies = {}
    for word in words:
        if word not in stop_words:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

    if not word_frequencies:
        return ' '.join(sentence_tokens[:2])

    max_frequency = max(word_frequencies.values())
    for word in word_frequencies:
        word_frequencies[word] = word_frequencies[word] / max_frequency

    sentence_scores = {}
    for idx, sent in enumerate(sentence_tokens):
        sent_words = re.findall(r'[a-zA-Z]{2,}', sent.lower())
        if not sent_words:
            continue
        score = 0.0
        for word in sent_words:
            if word in word_frequencies:
                weight = LEGAL_KEYWORDS.get(word, 1.0)
                score += word_frequencies[word] * weight
        # Length normalization to prevent run-on sentence bias
        norm_score = score / math.sqrt(len(sent_words))
        sentence_scores[idx] = norm_score

    select_length = max(1, min(len(sentence_tokens), max(2, int(len(sentence_tokens) * threshold))))
    top_indices = sorted(nlargest(select_length, sentence_scores, key=sentence_scores.get))
    summary = ' '.join([sentence_tokens[i] for i in top_indices])
    return summary


def run_benchmark():
    print("=" * 70)
    print("[*] Audited Benchmark: Ensuring F1 = 2 * P * R / (P + R)")
    print("=" * 70)

    dataset_path = "Dataset/tldrlegal_v1.json"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data_items = list(data.values())
    split_idx = min(69, int(len(data_items) * 0.82))
    test_data = data_items[split_idx:]
    num_samples = len(test_data)
    print(f"[+] Total samples: {len(data_items)} | Test evaluation split: {num_samples} samples")

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def avg(lst):
        return sum(lst) / max(1, len(lst))

    # 1. Profile Enhanced NLP Extractive Model
    print("\n[*] Profiling Enhanced NLP Extractive Model...")
    nlp_token_p, nlp_token_r = [], []
    nlp_r1_p, nlp_r1_r = [], []
    nlp_r2_p, nlp_r2_r = [], []
    nlp_rL_p, nlp_rL_r = [], []

    start_time_nlp = time.time()
    for item in test_data:
        text = item['original_text']
        ref = item['reference_summary']
        summary = summarize_extractive(text, threshold=0.35)

        tp, tr = calculate_token_metrics(ref, summary)
        nlp_token_p.append(tp)
        nlp_token_r.append(tr)

        score = scorer.score(ref, summary)
        nlp_r1_p.append(score['rouge1'].precision)
        nlp_r1_r.append(score['rouge1'].recall)
        nlp_r2_p.append(score['rouge2'].precision)
        nlp_r2_r.append(score['rouge2'].recall)
        nlp_rL_p.append(score['rougeL'].precision)
        nlp_rL_r.append(score['rougeL'].recall)

    nlp_total_time = time.time() - start_time_nlp
    nlp_latency_ms = round((nlp_total_time / num_samples) * 1000, 1)

    nlp_p = round(avg(nlp_token_p), 4)
    nlp_r = round(avg(nlp_token_r), 4)
    nlp_f = harmonic_f1(nlp_p, nlp_r)

    nlp_r1_p_val = round(avg(nlp_r1_p), 4)
    nlp_r1_r_val = round(avg(nlp_r1_r), 4)
    nlp_r1_f_val = harmonic_f1(nlp_r1_p_val, nlp_r1_r_val)

    nlp_r2_p_val = round(avg(nlp_r2_p), 4)
    nlp_r2_r_val = round(avg(nlp_r2_r), 4)
    nlp_r2_f_val = harmonic_f1(nlp_r2_p_val, nlp_r2_r_val)

    nlp_rL_p_val = round(avg(nlp_rL_p), 4)
    nlp_rL_r_val = round(avg(nlp_rL_r), 4)
    nlp_rL_f_val = harmonic_f1(nlp_rL_p_val, nlp_rL_r_val)

    nlp_results = {
        "model_name": "Enhanced NLP Extractive Summarizer",
        "architecture": "Frequency & Legal Cue Heuristic",
        "parameters": "Heuristic (Rule-based)",
        "latency_ms": nlp_latency_ms,
        "token_precision": nlp_p,
        "token_recall": nlp_r,
        "token_f1": nlp_f,
        "precision": nlp_p,
        "recall": nlp_r,
        "fscore": nlp_f,
        "rouge1_p": nlp_r1_p_val,
        "rouge1_r": nlp_r1_r_val,
        "rouge1_f": nlp_r1_f_val,
        "rouge2_p": nlp_r2_p_val,
        "rouge2_r": nlp_r2_r_val,
        "rouge2_f": nlp_r2_f_val,
        "rougeL_p": nlp_rL_p_val,
        "rougeL_r": nlp_rL_r_val,
        "rougeL_f": nlp_rL_f_val,
    }

    print(f"NLP Extractive -> P: {nlp_p:.4f} | R: {nlp_r:.4f} | F1: {nlp_f:.4f} (2PR/(P+R) verified)")
    print(f"ROUGE-1: {nlp_r1_f_val:.4f} | ROUGE-2: {nlp_r2_f_val:.4f} | ROUGE-L: {nlp_rL_f_val:.4f}")

    # 2. Profile Transformer (T5-Small)
    model_path = "./t5-small-offline"
    print(f"\n[*] Profiling Transformer (T5-Small) from '{model_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    param_count_str = f"{param_count / 1e6:.1f}M Parameters"

    t5_token_p, t5_token_r = [], []
    t5_r1_p, t5_r1_r = [], []
    t5_r2_p, t5_r2_r = [], []
    t5_rL_p, t5_rL_r = [], []

    start_time_t5 = time.time()
    with torch.no_grad():
        for item in test_data:
            input_text = "summarize: " + clean_legal_text(item['original_text'])
            ref_summary = clean_legal_text(item['reference_summary'])

            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            summary_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=150,
                min_length=25,
                num_beams=4,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                early_stopping=True
            )
            pred_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            tp, tr = calculate_token_metrics(ref_summary, pred_summary)
            t5_token_p.append(tp)
            t5_token_r.append(tr)

            score = scorer.score(ref_summary, pred_summary)
            t5_r1_p.append(score['rouge1'].precision)
            t5_r1_r.append(score['rouge1'].recall)
            t5_r2_p.append(score['rouge2'].precision)
            t5_r2_r.append(score['rouge2'].recall)
            t5_rL_p.append(score['rougeL'].precision)
            t5_rL_r.append(score['rougeL'].recall)

    t5_total_time = time.time() - start_time_t5
    t5_latency_ms = round((t5_total_time / num_samples) * 1000, 1)

    t5_p = round(avg(t5_token_p), 4)
    t5_r = round(avg(t5_token_r), 4)
    t5_f = harmonic_f1(t5_p, t5_r)

    t5_r1_p_val = round(avg(t5_r1_p), 4)
    t5_r1_r_val = round(avg(t5_r1_r), 4)
    t5_r1_f_val = harmonic_f1(t5_r1_p_val, t5_r1_r_val)

    t5_r2_p_val = round(avg(t5_r2_p), 4)
    t5_r2_r_val = round(avg(t5_r2_r), 4)
    t5_r2_f_val = harmonic_f1(t5_r2_p_val, t5_r2_r_val)

    t5_rL_p_val = round(avg(t5_rL_p), 4)
    t5_rL_r_val = round(avg(t5_rL_r), 4)
    t5_rL_f_val = harmonic_f1(t5_rL_p_val, t5_rL_r_val)

    t5_results = {
        "model_name": "Transformer (T5-Small) Summarizer",
        "architecture": "Seq2Seq Encoder-Decoder Transformer",
        "parameters": param_count_str,
        "latency_ms": t5_latency_ms,
        "token_precision": t5_p,
        "token_recall": t5_r,
        "token_f1": t5_f,
        "precision": t5_p,
        "recall": t5_r,
        "fscore": t5_f,
        "rouge1_p": t5_r1_p_val,
        "rouge1_r": t5_r1_r_val,
        "rouge1_f": t5_r1_f_val,
        "rouge2_p": t5_r2_p_val,
        "rouge2_r": t5_r2_r_val,
        "rouge2_f": t5_r2_f_val,
        "rougeL_p": t5_rL_p_val,
        "rougeL_r": t5_rL_r_val,
        "rougeL_f": t5_rL_f_val,
    }

    print(f"T5-Small      -> P: {t5_p:.4f} | R: {t5_r:.4f} | F1: {t5_f:.4f} (2PR/(P+R) verified)")
    print(f"ROUGE-1: {t5_r1_f_val:.4f} | ROUGE-2: {t5_r2_f_val:.4f} | ROUGE-L: {t5_rL_f_val:.4f}")

    benchmark_data = {
        "nlp_results": nlp_results,
        "t5_results": t5_results,
        "eval_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_sample_count": num_samples,
        "total_dataset_samples": len(data_items)
    }

    with open("eval_cache.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=4)
    print("\n[OK] Audited benchmark successfully saved to 'eval_cache.json'!")


if __name__ == "__main__":
    run_benchmark()
