# 📚 Unified RAG Pipeline for Multiple Data Sources

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-required-blue)](https://www.docker.com/)
[![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/dhruv25072003/RAG)

---

**This project implements a modular, extensible Retrieval-Augmented Generation (RAG) pipeline for interactive, AI-driven answers.**  
Supports ingestion from multiple data sources (structured & unstructured), seamless vector search (OpenSearch), and LLM-based summarization via a user-friendly Streamlit web app.

---

## 🚀 Key Features

- **Multi-Source Data Ingestion**  
  - Upload local files: `.csv`, `.txt`, `.md`, `.db`, or `.sqlite`
  - Connect directly to **Snowflake** or **PostgreSQL** warehouses/databases

- **Vector Search with OpenSearch**  
  - Fast, scalable, similarity-based retrieval of relevant chunks

- **LLM Summarization**  
  - Uses Hugging Face models (e.g. `flan-t5-base`) to generate concise, context-aware answers

- **Streamlit UI**  
  - Interactive sidebar for source/data selection
  - Guided interface for training and querying

- **Modular Python Package**  
  - `custom_rag` package structure for easy extension, adding new retrievers, data loaders, or models

---

## 📁 Project Structure

<details>
<summary>Show project layout</summary>

```
RAG/
│
├── app.py            # Main Streamlit app
├── requirements.txt  # Python dependencies
│
└── custom_rag/
    ├── __init__.py
    ├── constants.py
    ├── retriever.py        # Retriever classes (Composite, OpenSearch)
    ├── trainer.py          # HybridParentChildTrainer
    │
    ├── config/
    │   ├── __init__.py
    │   └── settings.py     # Mock settings
    │
    ├── dao/
    │   ├── __init__.py
    │   └── agent_dao.py    # Data access (mock/real)
    │
    ├── loaders/
    │   ├── __init__.py
    │   └── langchain_loader.py  # Multi-source loading (file, db, warehouse)
    │
    └── summarizer/
        ├── __init__.py
        └── summarizers.py  # Summarization classes (mock/real)
```
</details>

---

## ⚙️ Setup & Installation

1. **Clone this repository**
    ```
    git clone https://github.com/dhruv25072003/RAG.git
    cd RAG
    ```

2. **Create & activate your virtual environment**
    ```
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3. **Install dependencies**
    ```
    pip install -r requirements.txt
    ```

4. **Set up OpenSearch with Docker**
    ```
    docker run -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" --name opensearch-node -d opensearchproject/opensearch:2.11.0
    ```
    - Verify: Open [http://localhost:9200](http://localhost:9200) in your browser—should show a JSON welcome message

---

## ▶️ How To Run

```
streamlit run app.py
```
- This will launch your browser with the fully interactive RAG application UI.

---

## 📖 How To Use

1. **Select a Data Source:**  
   In the sidebar, pick "File Upload", "Snowflake", or "PostgreSQL".

2. **Provide Data:**  
   - File Upload: Choose `.csv`, `.db`, etc.
   - DB/Warehouse: Fill in credentials, table name, etc.

3. **Train the Model:**  
   - Hit **"Train from..."**.
   - The workflow:  
     - Loads & splits your data  
     - Generates vector embeddings  
     - Indexes all chunks in OpenSearch

4. **Ask a Question:**  
   - Enter questions in the main UI  
   - The system retrieves relevant context  
   - The LLM generates a summarized, natural language answer

5. **(Optional) Show Context:**  
   - Expand "Show Retrieved Context" to view raw passages used for the answer

---

## 🖼 Example UI

_The repository includes the existing interface. Please keep all branding, images, and screenshots intact as in the original app._

---

## 💡 Extending & Customizing

- Add new data loaders in `custom_rag/loaders/`
- Add more retriever types in `custom_rag/retriever.py`
- Swap or improve summarizers in `custom_rag/summarizer/`

---

## 📝 License
[MIT](LICENSE)

---

**Built with ❤️ by [dhruv25072003](https://github.com/dhruv25072003)**

_Star ⭐ this repo if you found it useful! PRs and feedback welcome!_
```

[1](https://github.com/dhruv25072003/RAG)
