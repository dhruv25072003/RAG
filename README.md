# Unified RAG Pipeline for Multiple Data Sources

This project implements a comprehensive Retrieval-Augmented Generation (RAG) pipeline that can connect to multiple data sources, process structured and unstructured data, and provide summarized, AI-driven answers to user queries. The application is built with Python and features an interactive web interface powered by Streamlit.

![App Screenshot](https://i.imgur.com/your-screenshot-url.png) 
*(Suggestion: Replace this with a screenshot of your running application, like the one you shared earlier)*

## 🚀 Features

* **Multi-Source Data Ingestion**: Connect to and process data from various sources:
    * **File Uploads**: `.csv`, `.txt`, `.md`, and SQLite (`.db`, `.sqlite`) files.
    * **Snowflake**: Directly connect to your Snowflake data warehouse.
    * **PostgreSQL**: Directly connect to your PostgreSQL database.
* **Vector Search**: Utilizes **OpenSearch** as a robust and scalable vector database to perform efficient similarity searches.
* **AI-Powered Summarization**: Employs a Hugging Face Language Model (`flan-t5-base`) to synthesize information retrieved from the data sources and provide concise, natural language answers.
* **Interactive Web UI**: A user-friendly interface built with **Streamlit** allows for easy data source selection, training, and querying.
* **Modular & Extensible**: The codebase is organized into a Python package (`custom_rag`), making it easy to extend with new data loaders, retrievers, or models.

## ⚙️ Project Structure

The project is organized into a Python package to ensure modularity and prevent import errors.

/RAG/||-- app.py                 # The main Streamlit application|-- requirements.txt       # Project dependencies||-- custom_rag/            # The core Python package|   |-- init.py|   |-- constants.py|   |-- retriever.py       # Contains retriever classes (Composite, OpenSearch)|   |-- trainer.py         # Contains the HybridParentChildTrainer class|   ||   |-- config/|   |   |-- init.py|   |   |-- settings.py    # Mock settings class|   ||   |-- dao/|   |   |-- init.py|   |   |-- agent_dao.py   # Mock Data Access Object classes|   ||   |-- loaders/|   |   |-- init.py|   |   |-- langchain_loader.py # Data loading logic for all sources|   ||   |-- summarizer/|       |-- init.py|       |-- summarizers.py # Mock summarizer classes
## 🛠️ Setup and Installation

Follow these steps to get the application running on your local machine.

### Prerequisites

* **Python 3.9+**
* **Docker Desktop**: Required to run the OpenSearch container.

### 1. Clone the Repository

```bash
git clone [https://github.com/dhruv25072003/RAG.git](https://github.com/dhruv25072003/RAG.git)
cd RAG
2. Set Up a Virtual EnvironmentIt is highly recommended to use a virtual environment.# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
3. Install DependenciesInstall all the required Python libraries from the requirements.txt file.pip install -r requirements.txt
4. Start OpenSearchRun a local OpenSearch instance using Docker. This command will download the image and start the container in the background.docker run -p 9200:9200 -p 9600:9600 -e "discovery.type=single-node" --name opensearch-node -d opensearchproject/opensearch:2.11.0
To verify that it's running, open http://localhost:9200 in your browser. You should see a JSON response.▶️ How to Run the ApplicationOnce the setup is complete, launch the Streamlit application from your terminal:streamlit run app.py
Your web browser will automatically open with the running application.📖 How to UseSelect a Data Source: Use the radio buttons in the sidebar to choose between "File Upload", "Snowflake", or "PostgreSQL".Provide Data:For File Upload, browse and select a supported file (.csv, .db, etc.).For Snowflake or PostgreSQL, fill in all the required connection credentials and the target table name.Train the Model: Click the "Train from..." button. The application will:Load the data from the selected source.Split the content into smaller chunks.Generate vector embeddings for each chunk.Create a new index in OpenSearch and ingest the data.Ask a Question: Once training is complete, the main area of the app will be enabled. Type your question into the text input and press Enter.Get the Answer: The application will retrieve the most relevant context from OpenSearch and use the LLM to generate a summarized answer. You can expand the "Show Retrieved Context" section to see the raw data that was used to create the answer.This project was developed with assistance from an AI model.</markdown>
