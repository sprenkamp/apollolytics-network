# Apollolytics Network Analysis

## Overview
This project analyzes Telegram data to understand the propagation of narratives across Russian and Ukrainian channels and groups. The workflow includes data extraction, narrative classification, message similarity analysis, and network visualization.

---

## Step-by-Step Analysis Flow

### 1. **Data Scraping (`scrape_telegram/`)**
- **Purpose:** Collect Telegram messages from Russian and Ukrainian channels and groups, and store them in PostgreSQL tables.
- **Key Scripts:**
  - `scrape_parallel_postgresSql_messages.py`: Main async scraper. Reads chat lists, scrapes messages using Telethon, and inserts them into categorized tables (e.g., `russian_channels_messages`, `ukrainian_groups_messages`). Handles batching, error logging, and schema creation.
  - `scrap_telegram_single_table.py`: Variant for scraping into a single table.
  - `scrape_tgstat.py` & `parse_tgstat.py`: Utilities for scraping and parsing channel/group metadata from tgstat.
- **Output:** Populated message tables in PostgreSQL, ready for cleaning and analysis.

### 2. **Basic Cleaning (`analysis/00_basic_clean/`)**
- **Purpose:** Initial cleaning and filtering of raw message data.
- **Key Scripts:**
  - `basic_remove.py`: Removes unwanted or malformed messages.
  - `basic_count_long.py`: Counts and analyzes long messages for further filtering.
- **Output:** Cleaned tables or CSVs for descriptive analysis.

### 3. **Descriptive Analysis (`analysis/01_descriptive/`)**
- **Purpose:** Generate descriptive statistics and summaries of the cleaned data.
- **Key Scripts:**
  - `descriptive_table_analysis.py`: Computes message counts, user activity, channel/group stats, and other descriptive metrics.
- **Output:** Summary tables, plots, and CSVs describing the dataset.

### 4. **Network Analysis (`analysis/02_network/`)**
- **Purpose:** Explore message forwarding and relationships between chats.
- **Key Scripts:**
  - `forwarded_message_network.py`: Builds and analyzes networks of message forwarding between chats/groups.
- **Output:** Network graphs and statistics on message propagation.

### 5. **Embeddings & Narrative Classification (`analysis/03_embedding/`)**
- **Purpose:** Generate message embeddings and classify messages by narrative.
- **Key Scripts:**
  - `01_embedding_gen.py`: Generates vector embeddings for messages.
  - `02_classify_messages_by_narrative.py`: Classifies messages into narratives using embeddings.
  - `03_narrative_similarity.py`: Calculates similarity scores for each message to each narrative.
- **Output:** Tables with narrative similarity scores for each message.

### 6. **Zero-Shot Classification (`analysis/04_zero_shot_classification/`)**
- **Purpose:** Apply zero-shot learning to verify or supplement narrative classification.
- **Key Scripts:**
  - `zero_shot_verificaiton.py`: Uses zero-shot models to check or enhance narrative labels.
- **Output:** Additional narrative labels or verification results.

### 7. **Narrative Analysis & Visualization (`analysis/05_analyse_narratives/`)**
- **Purpose:** Deep-dive into narrative propagation, sentiment, and network visualization.
- **Key Scripts:**
  - `01_narratives_overtime.py`: Tracks narrative prevalence over time.
  - `02_descriptive_analysis.py`: Further descriptive analysis focused on narratives.
  - `03_sentiment.py`: Sentiment analysis of messages.
  - `04_networks.py`: Builds interactive network graphs of narrative propagation (by country, combined, etc.).
  - `05_known_forwards.py`: Identifies and visualizes message forwards between known and unknown chats, with country/type coloring.
- **Output:** Interactive HTML network graphs, narrative time series, sentiment plots, and summary statistics.

### 8. **Jupyter Notebooks (`ipynb/`)**
- **Purpose:** Custom exploration, validation, and export of results.
- **Key Notebooks:**
  - `narrative_embeddings.ipynb`: Embedding analysis and narrative similarity exploration.
  - `verify_classification.ipynb`: Manual/visual verification of narrative classification.
  - `checkPostgres.ipynb`: Database checks and ad-hoc queries.
- **Output:** Custom tables, plots, and exports (CSV/Excel).

---

## Data Flow & Analysis Logic (Summary)

1. **Scrape Telegram data** →
2. **Clean and filter** →
3. **Descriptive stats** →
4. **Network analysis** →
5. **Embeddings & narrative classification** →
6. **Zero-shot verification** →
7. **Narrative/sentiment/network analysis** →
8. **Custom exploration/export in notebooks**

---

## Requirements
- Python 3.8+
- PostgreSQL
- Python packages: pandas, SQLAlchemy, python-dotenv, networkx, pyvis, matplotlib, seaborn, openpyxl, telethon, asyncpg, tqdm

## How to Run
- Set up your `.env` file with database credentials and Telegram API keys.
- Run scripts in the order above for a full pipeline, or use notebooks for custom analysis.
