"""
Fine-tuning Script for Legal/Financial Text Summarization using T5-Small
Adapted for B.Tech Final Year Project & Offline Inference.
"""

import os
import sys
import json
import time
import re

# Ensure Windows terminal outputs UTF-8 cleanly
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer


def clean_legal_text(text):
    """Clean legal document text for tokenizer processing."""
    if not text:
        return ""
    # Replace non-breaking spaces and unicode artifacts
    text = text.replace('\xa0', ' ').replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class LegalSummaryDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_input_len=512, max_target_len=128):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = "summarize: " + clean_legal_text(item['original_text'])
        target_text = clean_legal_text(item['reference_summary'])

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        targets = self.tokenizer(
            target_text,
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = targets.input_ids.squeeze(0)
        # Replace pad token id with -100 so cross-entropy ignores padding
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "labels": labels
        }


def fine_tune_and_export(
    dataset_path="Dataset/tldrlegal_v1.json",
    output_dir="./t5-small-offline",
    epochs=2,
    batch_size=2,
    lr=3e-4,
    cache_path="eval_cache.json"
):
    print("=" * 65)
    print("[*] Starting T5-Small Legal Document Fine-Tuning Pipeline")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")

    # 1. Load Dataset
    print(f"[*] Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data_raw = json.load(f)

    data_items = list(data_raw.values())
    total_samples = len(data_items)
    print(f"[+] Total samples loaded: {total_samples}")

    # Split train/test (70 train, 15 test)
    split_idx = min(70, int(total_samples * 0.82))
    train_data = data_items[:split_idx]
    test_data = data_items[split_idx:]
    print(f"[+] Train samples: {len(train_data)} | Test/Validation samples: {len(test_data)}")

    # 2. Load Model & Tokenizer
    model_name = "t5-small"
    print(f"[*] Initializing base model and tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    train_dataset = LegalSummaryDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 3. Fine-Tuning Loop
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()

    print("\n" + "-" * 40)
    print("[*] Training T5-Small on Legal TLDR Dataset...")
    print("-" * 40)

    start_time = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        step_count = 0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            step_count += 1

            if (step + 1) % 10 == 0 or (step + 1) == len(train_loader):
                print(f"  Epoch [{epoch+1}/{epochs}] Step [{step+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / max(1, step_count)
        print(f"--> Epoch {epoch+1} Completed. Average Loss: {avg_loss:.4f}\n")

    training_time = time.time() - start_time
    print(f"[+] Training completed in {training_time:.2f} seconds.")

    # 4. Save Fine-Tuned Model Locally
    print(f"[*] Exporting adapted weights & tokenizer to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[OK] Model successfully exported to '{output_dir}'. Offline inference ready!")

    # 5. Comprehensive Model Evaluation
    print("\n" + "=" * 65)
    print("[*] Evaluating Model Performance on Legal Test Set...")
    print("=" * 65)

    model.eval()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    t5_r1_p, t5_r1_r, t5_r1_f = [], [], []
    t5_r2_p, t5_r2_r, t5_r2_f = [], [], []
    t5_rL_p, t5_rL_r, t5_rL_f = [], [], []

    with torch.no_grad():
        for i, item in enumerate(test_data):
            input_text = "summarize: " + clean_legal_text(item['original_text'])
            ref_summary = clean_legal_text(item['reference_summary'])

            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(device)

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

            score = scorer.score(ref_summary, pred_summary)
            # ROUGE-1
            t5_r1_p.append(score['rouge1'].precision)
            t5_r1_r.append(score['rouge1'].recall)
            t5_r1_f.append(score['rouge1'].fmeasure)
            # ROUGE-2
            t5_r2_p.append(score['rouge2'].precision)
            t5_r2_r.append(score['rouge2'].recall)
            t5_r2_f.append(score['rouge2'].fmeasure)
            # ROUGE-L
            t5_rL_p.append(score['rougeL'].precision)
            t5_rL_r.append(score['rougeL'].recall)
            t5_rL_f.append(score['rougeL'].fmeasure)

    def avg(lst):
        return sum(lst) / max(1, len(lst))

    t5_results = {
        "precision": round(avg(t5_r1_p), 4),
        "recall": round(avg(t5_r1_r), 4),
        "fscore": round(avg(t5_r1_f), 4),
        "rouge1_p": round(avg(t5_r1_p), 4),
        "rouge1_r": round(avg(t5_r1_r), 4),
        "rouge1_f": round(avg(t5_r1_f), 4),
        "rouge2_p": round(avg(t5_r2_p), 4),
        "rouge2_r": round(avg(t5_r2_r), 4),
        "rouge2_f": round(avg(t5_r2_f), 4),
        "rougeL_p": round(avg(t5_rL_p), 4),
        "rougeL_r": round(avg(t5_rL_r), 4),
        "rougeL_f": round(avg(t5_rL_f), 4),
    }

    print("\n--- T5-Small Evaluated Metrics ---")
    print(f"Precision: {t5_results['precision']:.4f} | Recall: {t5_results['recall']:.4f} | F1: {t5_results['fscore']:.4f}")
    print(f"ROUGE-1 F1: {t5_results['rouge1_f']:.4f} | ROUGE-2 F1: {t5_results['rouge2_f']:.4f} | ROUGE-L F1: {t5_results['rougeL_f']:.4f}")

    # Save to eval_cache.json
    cache_data = {
        "t5_results": t5_results,
        "eval_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_sample_count": len(test_data),
        "training_time_sec": round(training_time, 2)
    }
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=4)
    print(f"[OK] Evaluation cache saved to '{cache_path}'.")

    return t5_results


if __name__ == "__main__":
    fine_tune_and_export()
