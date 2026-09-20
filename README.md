# NLP & Transformer-Based Legal and Financial Text Summarizer

An automated hybrid text summarization platform designed to distill lengthy, complex legal agreements, contracts, terms of service, and financial disclosures into actionable, concise summaries.

This project combines:
1. **Domain-Enhanced Extractive Summarization**: Fast keyword-boosted, sentence-scoring NLP engine tailored for legal cue phrases and operative clauses.
2. **Fine-Tuned Abstractive Transformer (T5-Small)**: A sequence-to-sequence neural model fine-tuned on the TLDRLegal contract dataset for human-like narrative synthesis.
3. **Dual Metric Evaluation**: Benchmarked using both multiset token overlap (Precision, Recall, F1) and standard ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L).
4. **Interactive Web Application**: A Django-powered interface with user authentication, document upload, real-time dual-engine summarization, and side-by-side comparison.

---

## Architecture Overview

```
                          ┌──────────────────────────────────────┐
                          │   Legal / Financial Input Document   │
                          └──────────────────┬───────────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
       ┌─────────────────────────────┐               ┌─────────────────────────────┐
       │     Extractive Engine       │               │     Abstractive Engine      │
       │   (Domain-Aware NLP)        │               │   (Fine-Tuned T5-Small)     │
       ├─────────────────────────────┤               ├─────────────────────────────┤
       │ • Text Cleaning & Splitting │               │ • Prefix: "summarize: "     │
       │ • Legal Cue Weighting       │               │ • Subword Tokenization      │
       │ • Length-Normalized Scoring │               │ • Self- & Cross-Attention   │
       │ • Chronological Assembly    │               │ • Beam Search Decoding (k=4)│
       └──────────────┬──────────────┘               └──────────────┬──────────────┘
                      │                                             │
                      ▼                                             ▼
       ┌─────────────────────────────┐               ┌─────────────────────────────┐
       │ Extractive Summary          │               │ Abstractive Summary         │
       │ (Verbatim Contract Clauses) │               │ (Concise AI Synthesis)      │
       └──────────────┬──────────────┘               └──────────────┬──────────────┘
                      │                                             │
                      └──────────────────────┬──────────────────────┘
                                             ▼
                              ┌─────────────────────────────┐
                              │ Comparative Evaluation View │
                              │ (ROUGE-1/2/L, F1, Latency)  │
                              └─────────────────────────────┘
```

---

## Project Structure

```
NLP-Legal-Financial-Summarizer/
├── Dataset/
│   └── tldrlegal_v1.json         # TLDRLegal contract dataset (JSON format)
├── Summary/                      # Django project configuration
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── SummaryApp/                   # Main Django application
│   ├── static/                   # CSS styles and UI images
│   ├── templates/                # HTML templates (Login, Signup, Summarizer, etc.)
│   ├── models.py
│   ├── urls.py
│   └── views.py                  # Core summarization logic & inference controllers
├── t5-small-offline/             # Fine-tuned T5-Small offline Transformer model
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors         # 230+ MB Model weights (tracked via Git LFS)
│   ├── special_tokens_map.json
│   ├── spiece.model
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── fine_tune_t5.py               # Fine-tuning & offline export script
├── test_models.py                # Standalone evaluation & model benchmarking suite
├── verify_views.py               # Django view logic verification script
├── nltkdownload.py               # Automated NLTK dependency downloader
├── download_nltk.bat             # Windows script to launch NLTK GUI downloader
├── manage.py                     # Django management CLI
├── req.txt                       # Python dependencies list
├── run.bat                       # Quick-launch batch file for Django dev server
├── DB.txt                        # MySQL database schema and setup queries
├── DatasetLink.txt               # Source reference for TLDRLegal dataset
├── instructions.txt              # Environment setup guide
├── PROJECT_REPORT.md             # Full B.Tech Major Project Academic Report
├── SCREENS.docx                  # Application screenshots documentation
├── Modified_screens.docx         # Updated UI screenshots
└── README.md
```

---

## Transformer Model Distribution & Git LFS

The fine-tuned T5-Small model weights (`t5-small-offline/model.safetensors`, ~230 MB) are managed in this repository using **Git Large File Storage (Git LFS)**.

### Option 1: Using Git LFS (Recommended)

1. Ensure Git LFS is installed on your machine:
   ```bash
   git lfs version
   ```
   *(If not installed, download it from [git-lfs.com](https://git-lfs.com/) or run `winget install GitHub.GitLFS` on Windows).*

2. Clone the repository and fetch the LFS objects:
   ```bash
   git clone https://github.com/madhurireddygurram/NLP-Legal-Financial-Summarizer.git
   cd NLP-Legal-Financial-Summarizer
   git lfs pull
   ```

### Option 2: Local Regeneration / Training (Fallback)

If Git LFS is unavailable or if you prefer to build/fine-tune the model directly from the dataset:
```bash
python fine_tune_t5.py
```
This script downloads base `t5-small` weights from HuggingFace, fine-tunes them on `Dataset/tldrlegal_v1.json`, and exports the model files directly into `./t5-small-offline/`.

---

## Installation & Setup

### 1. Prerequisites
- Python 3.7 to 3.9 (recommended)
- Git & Git LFS
- MySQL Server (or XAMPP / WAMP)

### 2. Clone the Repository
```bash
git clone https://github.com/madhurireddygurram/NLP-Legal-Financial-Summarizer.git
cd NLP-Legal-Financial-Summarizer
git lfs pull
```

### 3. Install Python Dependencies
```bash
pip install -r req.txt
```

### 4. Download Required NLTK Corpora
Run the automated NLTK downloader:
```bash
python nltkdownload.py
```
*(Alternatively, execute `download_nltk.bat`)*

### 5. Configure MySQL Database
1. Open your MySQL client (e.g. MySQL Command Line or phpMyAdmin).
2. Execute the schema statements found in `DB.txt`:
   ```sql
   CREATE DATABASE summary;
   USE summary;

   CREATE TABLE signup(
       username VARCHAR(50),
       password VARCHAR(50),
       contact_no VARCHAR(12),
       email_id VARCHAR(50),
       address VARCHAR(50)
   );
   ```
3. Update MySQL database credentials in `Summary/settings.py` (and `SummaryApp/views.py`) if your local MySQL user/password differs from `root`/empty.

### 6. Run Migrations & Start Server
```bash
python manage.py makemigrations
python manage.py migrate
python manage.py runserver
```
*(Or simply double-click `run.bat` on Windows)*

Access the application in your web browser at:
```
http://127.0.0.1:8000/
```

---

## Model Evaluation & Testing

You can evaluate the models independently without running the web server:

```bash
# Test inference & evaluate ROUGE and F1 scores
python test_models.py

# Verify view processing & summarizer logic
python verify_views.py
```

---

## Performance Summary

| Metric | Extractive Summarizer | Fine-Tuned T5-Small |
| :--- | :--- | :--- |
| **Model Type** | NLP Frequency & Domain-Cue Engine | Seq2Seq Transformer (60.5M params) |
| **Average Inference Latency** | ~3.6 ms / document | ~1,313 ms / document (CPU) |
| **Summary Style** | Key Operative Verbatim Clauses | Abstractive, Natural Language Synthesis |
| **ROUGE-1 F1** | ~0.354 | ~0.428 |
| **ROUGE-2 F1** | ~0.158 | ~0.216 |
| **ROUGE-L F1** | ~0.282 | ~0.374 |

---

## License

This project is developed for educational and research purposes under the B.Tech Major Project curriculum.
