# Nokia_DBLP_dataset_analysis

> **Nokia Data Engineer – Summer Trainee Recruitment Task**  
> Comprehensive analysis of the DBLP bibliographic dataset using classical NLP, modern sentence embeddings, semantic search, RAG, and supervised ML.

---

## Overview

This project analyses the [DBLP Computer Science Bibliography](https://dblp.org/) – a dataset of over 6 million CS publications – with the goal of extracting as much insight as possible. Due to hardware constraints, the analysis uses a **subset of ~961,027 records** (years up to 2024), covering conference papers, book chapters, books, and proceedings.

### What's inside

| Section | Description |
|---|---|
| **1. Schema Exploration** | Empirical XML schema discovery via streaming parse |
| **2. Final Data Parse** | Targeted extraction of 1 M records → Parquet |
| **3. EDA** | Publication volume, author behaviour, page counts, venue analysis, title keyword trends |
| **4. Topic Clustering** | TF-IDF + KMeans vs. Sentence Embeddings + KMeans, UMAP visualisation |
| **5. Semantic Search** | Dense vector retrieval over 150 K paper titles |
| **6. RAG Pipeline** | Natural-language Q&A grounded in retrieved DBLP papers (local LLM via Ollama) |
| **7. Classification** | Supervised prediction of publication type from title text |

---

## Repository Structure

```
Nokia_DBLP_dataset_analysis/
│
├── Nokia_DBLP_analysis_final.ipynb     # Main Jupyter Notebook (with outputs)
├── Nokia_DBLP_analysis_final_pdf.pdf   # PDF export of the notebook for quick preview
│                                        # (the .ipynb is too large to preview on GitHub)
├── requirements.txt                     # Python dependencies
├── .gitignore
│
├── figures/                             # All saved visualisation figures
│   ├── fig01_publications_per_year.png
│   ├── fig02_top_authors_trend.png
│   ├── fig03_team_size_trends.png
│   ├── fig04_collaboration_network.png
│   ├── fig05_page_count_distribution.png
│   ├── fig06_page_count_by_author.png
│   ├── fig07_page_count_over_time.png
│   ├── fig08_top_venues_over_time.png
│   ├── fig09_venue_scatter.png
│   ├── fig10_author_venue_heatmap.png
│   ├── fig11_keyword_trends_historical.png
│   ├── fig12_keyword_trends_recent.png
│   ├── fig13_elbow_tfidf.png
│   ├── fig14_cluster_bar_k20.png
│   ├── fig15_cluster_bar_k20_all.png
│   ├── fig16_cluster_bar_k20_excluded.png
│   ├── fig17_cluster_bar_k7.png
│   ├── fig18_umap_tfidf_k7.png
│   ├── fig19_umap_tfidf_lsa_k7.png
│   ├── fig20_silhouette_tfidf.png
│   ├── fig21_umap_embeddings_k7.png
│   ├── fig22_silhouette_emb.png
│   ├── fig23_confusion_matrix.png
│   ├── fig24_feature_importance.png
│   ├── umap_tfidf_k7.html              # Interactive Plotly UMAP (TF-IDF)
│   ├── umap_tfidf_lsa_k7.html          # Interactive Plotly UMAP (TF-IDF + LSA)
│   └── umap_embed_k7.html              # Interactive Plotly UMAP (embeddings)
│
└── json files/                          # Cached intermediate results
    ├── dblp_sample_cache.json           # XML schema structure scan
    ├── cluster_labels_k20.json          # LLM-generated cluster labels (k=20)
    ├── cluster_labels_k7.json           # LLM-generated cluster labels (k=7, TF-IDF)
    ├── cluster_labels_emb.json          # LLM-generated cluster labels (embeddings)
    └── silhouette_results.json          # Cached silhouette scores (TF-IDF)
```

---

## Large Files (GitHub Release v1.0)

The following files are **too large for the repository** and are provided in the [**v1.0 Release**](../../releases/tag/v1.0):

| File | Size | Description |
|---|---|---|
| `dblp_dataset_initial.parquet` | 429 MB | Initial 4 M record parse (all 10 record types) |
| `dblp_dataset_final.parquet` | 106 MB | Final 961 K record parse (5 focused record types, ≤2024) |
| `embeddings_dblp.npy` | 220 MB | Sentence embeddings for 150 K paper titles (all-MiniLM-L6-v2) |

**Download these files and place them in the same directory as the notebook before running it.** The notebook checks for these files on startup and skips the (slow) parse/embed steps if they are present.

---

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/zofiawarzycha/Nokia_DBLP_dataset_analysis.git
cd Nokia_DBLP_dataset_analysis
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the DBLP dataset

The official DBLP XML can be downloaded from [https://dblp.org/xml/](https://dblp.org/xml/).  
You will need both `dblp.xml` and `dblp.dtd`.

Place both files in a location of your choice and set the path via environment variable:

```bash
# Linux / macOS
export DBLP_PATH=/path/to/dblp.xml

# Windows CMD
set DBLP_PATH=C:\path\to\dblp.xml
```

Alternatively, place `dblp.xml` and `dblp.dtd` in the same directory as the notebook (fallback).

### 4. Download pre-processed data (recommended)

Download the three Parquet/NPY files from the [v1.0 Release](../../releases/tag/v1.0) and place them next to the notebook. This allows you to skip the time-consuming parse and embedding steps entirely.

### 5. Set up Ollama (for RAG and LLM cluster labelling)

The RAG pipeline and LLM-assisted cluster labelling require [Ollama](https://ollama.com/) running locally.

```bash
# Install Ollama from https://ollama.com
ollama pull llama3.2
# Make sure Ollama is running before executing those cells
```

If Ollama is not available, the cluster-labelling function falls back to manually defined labels (included in the notebook), and the RAG cells will return an informative message instead of failing.

---

## Running the Notebook

```bash
jupyter notebook Nokia_DBLP_analysis_final.ipynb
```

Each section has a **section guard cell** at the top that loads the required DataFrames from Parquet if they are not already in memory – allowing you to jump directly to any section without re-running earlier cells.

---

## Key Results Summary

| Dimension | Key Finding |
|---|---|
| **Publication volume** | 54 335 records in 2024 – highest in the dataset |
| **Authorship** | Average authors/paper rose from 2.0 (1985) to 3.5+ (2024) |
| **Top author** | H. Vincent Poor – 385 publications, career since 1994 |
| **Top venue** | ICASSP – 44 341 papers (avg 4.49 pages) |
| **Longest papers** | ACL (14.53 pages avg), SODA (13.33), NeurIPS (12.04) |
| **LLM boom** | "language" entered global top-3 title keywords in 2024 |
| **Clustering** | Sentence embeddings give clear UMAP separation; TF-IDF does not |
| **Classification** | Logistic Regression: macro F1 = 0.74 (title-only, balanced classes) |

---

## Notes on Dataset Coverage

The analysis uses approximately **1 M out of ~6 M total DBLP records** (the first portion of the XML file). This subset is dominated by `inproceedings` (~884 K) and contains very few `article` records (~12), since journal articles appear later in the XML. The 1 M cap was chosen to balance analytical depth with hardware constraints on a standard laptop.
