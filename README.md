# 🧠 Enterprise RAG System

An enterprise-grade Retrieval-Augmented Generation (RAG) system that enables accurate question answering over private PDF documents using semantic search, vector embeddings, and Large Language Models.

Instead of retraining an LLM on proprietary data, this project implements the industry-standard RAG architecture, allowing users to query internal documents while minimizing hallucinations and ensuring responses are grounded in retrieved context.

---

## 🚀 Features

* 📄 PDF document ingestion and text extraction
* 🧹 NLP-based text cleaning and preprocessing
* ✂️ Context-aware chunking with sliding window overlap
* 🔍 Semantic search using vector embeddings
* 🗄️ FAISS-powered vector database for efficient retrieval
* 🤖 LLM-powered answer generation using Qwen2.5
* 🎯 Top-K document retrieval for context-grounded responses
* ⚡ Interactive command-line question answering interface

---

## 🏗️ System Architecture

```text
PDF Document
      │
      ▼
Text Extraction
      │
      ▼
Text Cleaning
      │
      ▼
Chunking
(Sliding Window)
      │
      ▼
Embeddings
(all-MiniLM-L6-v2)
      │
      ▼
FAISS Vector Database
      │
      ▼
Top-K Retrieval
      │
      ▼
Prompt Construction
      │
      ▼
Qwen2.5 LLM
      │
      ▼
Generated Answer
```

---

## 📂 Project Structure

```text
Enterprise-RAG-System/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── vector_db/
│
├── notebooks/
│
├── scripts/
│
├── src/
│   ├── ingestion/
│   ├── embedding/
│   ├── retrieval/
│   ├── generation/
│   ├── pipeline/
│   └── vectorstore/
│
├── tests/
│
├── config.yaml
├── requirements.txt
└── main.py
```

---

## 🛠️ Technology Stack

### NLP & Document Processing

* Python
* PyPDF2 / PDF Processing
* LangChain Text Splitters

### Embeddings

* Sentence Transformers
* all-MiniLM-L6-v2

### Vector Database

* FAISS (Facebook AI Similarity Search)

### Large Language Model

* Qwen2.5-7B-Instruct

### Machine Learning Framework

* PyTorch
* Hugging Face Transformers

---

## ⚙️ Configuration

The project configuration is managed through `config.yaml`.

```yaml
embedding:
  model_name: "all-MiniLM-L6-v2"
  batch_size: 32

chunking:
  chunk_size: 400
  chunk_overlap: 100

retrieval:
  top_k: 3

llm:
  model: "Qwen/Qwen2.5-7B-Instruct"
  max_tokens: 1024

paths:
  raw_data: "data/raw/"
  processed_data: "data/processed/"
  vector_db: "data/vector_db/"
```

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Enterprise-RAG-System.git
cd Enterprise-RAG-System
```

### 2. Create Virtual Environment

```bash
python -m venv env_rag
```

### 3. Activate Virtual Environment

#### Windows

```bash
env_rag\Scripts\activate
```

#### Linux / macOS

```bash
source env_rag/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Start the application:

```bash
python main.py
```

Provide the path to your PDF document when prompted:

```text
📄 Enter the absolute path to your PDF document:
```

The system will:

1. Ingest and process the document
2. Generate embeddings
3. Create a FAISS vector index
4. Launch an interactive Q&A session

Example:

```text
❓ Your Question:
What is the company's leave policy?
```

---

## 🔄 Commands

During interactive mode:

```text
quit      → Exit application
reingest  → Rebuild vector database
```

---

## 📊 Retrieval-Augmented Generation Workflow

### Stage 1: Document Ingestion

* Load PDF documents
* Extract raw text
* Clean and normalize content
* Generate overlapping chunks

### Stage 2: Vectorization

* Generate embeddings using Sentence Transformers
* Store vectors in FAISS index

### Stage 3: Retrieval

* Convert user query into embedding
* Retrieve Top-K relevant chunks

### Stage 4: Generation

* Construct context-aware prompt
* Generate answer using Qwen2.5
* Return grounded response

---

## 🎯 Example Use Cases

* Enterprise Knowledge Management
* Internal HR Policy Assistant
* Technical Documentation Search
* Customer Support Knowledge Base
* Compliance and Regulatory Document Search
* Research Paper Question Answering

---

## 📈 Future Improvements

* Web-based user interface
* Multi-document support
* Hybrid search (BM25 + Dense Retrieval)
* Metadata filtering
* Conversation memory
* Citation-aware responses
* Evaluation framework for retrieval quality
* Cloud deployment using Docker

---

## 👨‍💻 Author

Ayush Pradhan

Industrial Project: Architecting RAG Systems for Enterprise Data

---

## 📄 License

This project is intended for educational and research purposes.

