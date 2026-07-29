<div align="center">

# StructRAG
**Hybrid Retrieval-Augmented Generation (RAG) System**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

*Precision answer generation across complex financial and academic documents, backed by structured citations and high-fidelity multi-modal extraction.*

</div>

---

## Overview

Standard Retrieval-Augmented Generation (RAG) pipelines break down when faced with structurally dense PDFs, multi-page financial tables, and domain-specific acronyms. 

**StructRAG** is purposely built to solve these challenges. It introduces a multi-backend parsing engine, semantic hierarchical chunking, and parallel dense-sparse (Hybrid) retrieval fused with Reciprocal Rank Fusion (RRF). Together with triple-layer background verification, it guarantees responses that are strictly grounded, mathematically verified, and meticulously cited.

## Key Features

- **Multi-Modal Parsing:** Dynamically routes document pages to the optimal extraction engine (**PyMuPDF** for digital, **pdfplumber** for tables, **Docling** for OCR).
- **Hierarchical Chunking:** Implements parent-child semantic chunking, naturally preserving token limits while preventing context fragmentation. Special handling for table rows and abbreviation-safe sentence splitting.
- **Hybrid Retrieval (Dual-Vector):** Parallel search queries over **Qdrant** using both Dense vectors (OpenAI `text-embedding-3-large`) and Sparse learned embeddings (local **SPLADE** model), fused via RRF.
- **High-Fidelity Reranking:** Leverages an external CrossEncoder microservice and custom numeric boosting (2x weight on exact number matches) to surface only the most contextually perfect chunks.
- **Triple-Verified Generation:** Asynchronous streaming LLM (`gpt-4o-mini`) is immediately followed by background checks for: (1) Numeric accuracy, (2) Claim grounding & hallucination, (3) Citation formatting. 

---

## Architecture

```mermaid
flowchart TB
    %% Ingestion Phase
    subgraph Ingestion["Ingestion Phase (Offline)"]
        direction TB
        PDF["PDF File"] --> Profiler{"Page Profiler"}
        
        Profiler -- "Low Text/Images" --> Docling["Docling (Scanned / OCR)"]
        Profiler -- "Tables Detected" --> Plumber["pdfplumber (Tables)"]
        Profiler -- "Standard Text" --> PyMuPDF["PyMuPDF (Digital Text)"]
        
        Docling --> Merge["Merge & Normalize Text"]
        Plumber --> Merge
        PyMuPDF --> Merge
        
        Merge --> Chunker["chunker.py<br/>(Parent / Child Chunks)"]
        Chunker --> Embedder["embedder.py<br/>(Dense + Sparse Vectors)"]
        Embedder --> DB[("Qdrant Vector DB<br/>(database.py)")]
    end

    %% Query Phase
    subgraph Query["Query Phase (Online)"]
        direction TB
        UserQuery["User Query"] --> QueryEmbed["embedder.py<br/>(Dense + Sparse)"]
    end

    %% Retrieval Phase
    subgraph Retrieval["Retrieval Process"]
        direction TB
        Search["database.py<br/>(Hybrid Search + RRF)"] --> Retrieve["retriever.py<br/>(Rerank + Filter + Dedup)"]
        Retrieve --> Context["Context Assembly<br/>(Child -> Parent text)"]
    end

    %% Generation Phase
    subgraph Generation["Generation Process"]
        direction TB
        Gen["generator.py<br/>(GPT-4o-mini stream)"] --> Verify["3 Async Verifiers<br/>(Facts - Numbers - Citations)"]
        Verify --> Output["Answer + Confidence Score"]
    end

    %% Cross-phase connections
    DB -. "Stored Vectors" .-> Search
    QueryEmbed --> Search
    UserQuery -.-> Gen
    Context -- "Assembled Context" --> Gen
```

### 1. Ingestion Phase (Offline)
StructRAG builds a comprehensive representation of your documents before any queries are asked:
- **Intelligent Page Profiling:** As PDFs enter the system, a fast `Page Profiler` evaluates text density and layout to determine the optimal processing path.
- **Multi-Modal Routing:** Pages are dynamically routed to: 
  - **Docling:** For robust OCR on scanned images and low-text pages.
  - **pdfplumber:** Specifically targets and extracted tabular structures into Markdown format.
  - **PyMuPDF:** Efficiently parses standard digital text and metadata.
- **Hierarchical Chunking & Embedding:** The merged texts are parsed by `chunker.py`, creating broad **Parent** chunks for context retention and granular **Child** chunks for high-precision retrieval. The `embedder.py` generates dual representations: traditional Dense embeddings via OpenAI, and learned Sparse expansions via a local SPLADE model. These are persisted to the **Qdrant Vector Database**.

### 2. Query Phase (Online)
When a user asks a question, the input passes through the identical dual-encoding process (`embedder.py`), transforming the plain text into Dense semantic vectors and Sparse exact-match term expansions.

### 3. Retrieval Process
- **Hybrid Search & Fusion:** The system queries the Qdrant DB with both vector types simultaneously. Results are mathematically merged using Reciprocal Rank Fusion (RRF) in `database.py`.
- **Rerank & Filter:** Candidates enter `retriever.py` where a CrossEncoder AI model individually scores and reranks the relevance of every retrieved chunk, while deduplication mechanisms strip redundant data.
- **Context Assembly:** The highly-ranked Child chunks retrieve their extensive Parent chunk's text, assembling a rich, surrounding context neighborhood for the LLM.

### 4. Generation Process
- **Assembled Context Generation:** `generator.py` streams the user query and the expanded context into an LLM (`gpt-4o-mini`).
- **Async Verification:** While generating, three background workers aggressively audit the response:
  - **Fact Check:** Dissects the output to ensure each atomic fact is supported by the context.
  - **Number Match:** Verifies that all numeric outputs exactly mirror figures inside the original document.
  - **Citation Check:** Confirms robust `[Source X | Page Y]` attribution formatting.
- **Calibrated Output:** The final response arrives alongside a fused Confidence Score summarizing the outcome of all validation checks.

---

## Getting Started

### Prerequisites
- Python 3.12+
- OpenAI API Key

### 1. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/your-username/StructRAG.git
cd StructRAG

# Create and activate virtual environment
python3.12 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `key.env` file in the project root to securely store your variables:

```env
OPENAI_API_KEY="sk-your-openai-api-key"
# Optional: Setup Qdrant DB for cloud mode
# QDRANT_URL="https://your-cluster-url.qdrant.tech"
# QDRANT_API_KEY="your-qdrant-api-key"
```

### 3. Usage

StructRAG runs out-of-the-box in local mode. Start the ingestion and interactive REPL loop by passing a directory of PDFs or a single file:

```bash
python main.py /path/to/your/documents/
```

**Interactive REPL Commands:**
- `Query>` Simply type your query and press enter (e.g. `What was the total revenue for FY2023?`).
- `clear` - Clears the terminal screen.
- `quit` or `exit` - Shuts down the system gracefully.

---

## Evaluation & Benchmarking

The system is continuously tested against datasets like **FinanceBench** using the industry-standard **RAGAS** framework. 

To run the benchmarking suite:
```bash
python evaluate.py
```
This produces a detailed CSV report judging the pipeline on four critical metrics:
- **Context Precision:** Are the most relevant chunks ranked at the top?
- **Context Recall:** Did the system retrieve all the context truly needed to answer?
- **Faithfulness:** Is the generated answer hallucination-free?
- **Answer Relevancy:** How well does the answer address the user's specific query?

---

## Repository Structure

```text
StructRAG/
├── main.py              # Application entry point & REPL loop
├── pdf_parser.py        # Intelligent routing to PyMuPDF, pdfplumber, Docling
├── chunker.py           # Configurable parent-child semantic splitting
├── embedder.py          # OpenAI Dense & SPLADE Sparse vectorization 
├── database.py          # Dual-index chunk storage with Qdrant
├── retriever.py         # Search, RRF, Rerank, & metadata filtering logic 
├── generator.py         # LLM abstraction with citation constraints
├── evaluate.py          # RAGAS evaluation harness
├── config.py            # Global hyperparameter management
└── models.py            # Pydantic data schemas 
```

---

## Contributing

Contributions are welcome! If you have ideas for architectural improvements or feature requests:
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

---
<div align="center">
  <i>Built for high-stakes document intelligence.</i>
</div>
