# Manava Analytics

A personal AI-powered data analytics platform that runs **100% locally** on your machine. Upload any dataset and get instant descriptive stats, correlation analysis, forecasting, and clustering — with one-click exports to Excel, PowerPoint, and PDF.

---

## Features

- Upload CSV / TSV files or paste data directly
- Auto-runs 4 types of analysis on any dataset
- Interactive dashboard with charts
- Ask AI questions about your data (powered by Ollama)
- Export to Excel (.xlsx), PowerPoint (.pptx), and PDF
- Fully local — no API keys, no cloud, no cost

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML + Chart.js |
| AI Chat | Ollama (local LLM) |
| Analytics | Python + Flask + pandas + scikit-learn |
| Excel export | openpyxl |
| PPT export | PptxGenJS (Node.js) |
| PDF export | ReportLab |

---

## Setup

### Requirements
- macOS (tested on Apple M1)
- Python 3.10+
- Node.js 18+
- Ollama

### 1. Clone the repo
```bash
git clone https://github.com/manureddy04/Manava-AI_analytics.git
cd Manava-AI_analytics
```

### 2. Install Python dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install flask flask-cors pandas numpy scikit-learn openpyxl python-pptx reportlab
```

### 3. Install Node dependencies
```bash
npm install pptxgenjs
```

### 4. Install and set up Ollama
```bash
# Download from https://ollama.com/download
ollama pull llama3.2
```

---

## Running the App

Open **3 terminal tabs** and run one command in each:

**Tab 1 — Ollama:**
```bash
OLLAMA_ORIGINS="*" OLLAMA_HOST="0.0.0.0:11434" ollama serve
```

**Tab 2 — Analytics server:**
```bash
cd Manava-AI_analytics
source venv/bin/activate
python3 server.py
```

**Tab 3 — Web server:**
```bash
cd Manava-AI_analytics
python3 -m http.server 8080
```

Then open Chrome → **http://localhost:8080**

---

## Usage

1. Drag & drop a CSV file or paste data into the text box
2. Manava auto-runs all analyses and shows the dashboard
3. Use the tabs to switch between Overview, Descriptive, Correlation, Forecast, Clustering
4. Click **Excel**, **PPT**, or **PDF** to export reports
5. Click **Ask AI** to ask plain English questions about your data

---

## Project Structure

```
Manava-AI_analytics/
├── index.html       # Frontend dashboard
├── server.py        # Flask analytics backend
├── make_pptx.js     # PowerPoint generator (Node.js)
├── setup.sh         # One-time setup script
└── .gitignore
```

---

## Analysis Types

| Type | What it does |
|------|-------------|
| Descriptive Stats | Mean, median, std dev, skew, min, max for all numeric columns |
| Correlation | Pearson r between all column pairs, scatter plot of strongest pair |
| Forecasting | Exponential smoothing to predict future values |
| Clustering | K-Means segmentation with PCA 2D visualization |

---

Built by [@manureddy04](https://github.com/manureddy04)
