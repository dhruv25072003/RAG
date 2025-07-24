import streamlit as st
import os
import tempfile
from opensearchpy import OpenSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Any, Dict, Tuple, List

# --- Correctly import from the custom_rag package ---
from custom_rag.trainer import HybridParentChildTrainer
from custom_rag.retriever import Retriever, CompositeRetriever, OpenSearchRetriever
from custom_rag.config.settings import KnowledgeSettings
from custom_rag.dao.agent_dao import AgentFileInfoIndex, AgentMetaIndex

# --- App Configuration ---
st.set_page_config(page_title="Custom RAG Pipeline", layout="wide")
st.title("RAG Pipeline with Custom Trainer and Retriever")

# --- Global Settings ---
OPENSEARCH_URL = "http://localhost:9200"
DATA_INDEX_NAME = "my-final-rag-index"
AGENT_ID = "test-agent-001"
TENANT_NAME = "default-tenant"
USER_ID = "streamlit-user"

# --- Concrete OpenSearch Index Class ---
class OpenSearchIndex:
    """A concrete class to manage the OpenSearch index."""
    def __init__(self, client, index_name):
        self.client = client
        self.index_name = index_name

    def create(self):
        """Deletes the old index and creates a new one with the correct k-NN mapping."""
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
        
        index_body = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "embeddings": {
                        "type": "knn_vector",
                        "dimension": 384  # Dimension for all-MiniLM-L6-v2
                    },
                    "text": {"type": "text"},
                    "source": {"type": "keyword"}
                }
            }
        }
        # FIX: Corrected the call to client.indices.create with keyword arguments
        self.client.indices.create(index=self.index_name, body=index_body)

    def bulk_addrecords(self, records: List[Document]):
        """Indexes a list of documents into OpenSearch."""
        for doc in records:
            self.client.index(
                index=self.index_name,
                body={
                    "text": doc.page_content,
                    "embeddings": doc.metadata['embeddings'],
                    "source": doc.metadata.get('source')
                },
                refresh=True
            )

# --- Main App Logic ---
@st.cache_resource
def get_global_resources():
    """Initializes and caches global resources like the DB client and embedding model."""
    try:
        client = OpenSearch(OPENSEARCH_URL, timeout=30)
        if not client.ping():
            raise ConnectionError("Could not connect to OpenSearch.")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        return client, embeddings
    except Exception as e:
        st.error(f"Failed to initialize resources: {e}")
        return None, None

client, embeddings = get_global_resources()

if 'retriever' not in st.session_state:
    st.session_state.retriever = None

if not client:
    st.error("Application cannot start. Please ensure OpenSearch is running and accessible at " + OPENSEARCH_URL)
else:
    with st.sidebar:
        st.header("File Upload and Training")
        uploaded_file = st.file_uploader("Upload a file", type=["txt", "md", "csv"])

        if st.button("Train on Uploaded File"):
            if uploaded_file:
                with st.status("Training in progress...", expanded=True) as status:
                    try:
                        with tempfile.TemporaryDirectory() as temp_dir:
                            input_path = os.path.join(temp_dir, AGENT_ID)
                            os.makedirs(input_path, exist_ok=True)
                            file_path = os.path.join(input_path, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())

                            status.write("Initializing trainer and dependencies...")
                            
                            trainer = HybridParentChildTrainer(
                                es=client, agent_id=AGENT_ID, agent_type="test", tenant_name=TENANT_NAME,
                                agent_name="Test Agent", settings=KnowledgeSettings(temp_dir),
                                embedding_obj=embeddings, data_index=OpenSearchIndex(client, DATA_INDEX_NAME),
                                meta_index=AgentMetaIndex(),
                                user_id=USER_ID, suggestions=[], avatar_color="#FFF",
                                configurations={'chunk_sizes': {'parent': 1000, 'child': 300}}
                            )
                            
                            status.write("Loading and chunking document...")
                            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                            docs, _, _, _, _ = trainer.loader.load_and_convert(folder_path=input_path, shortlisted_files=[file_path])
                            chunks = text_splitter.split_documents(docs)
                            status.write(f"Split document into {len(chunks)} chunks.")
                            
                            status.write("Generating embeddings...")
                            chunks_with_embeddings = trainer.get_embeddings(chunks)
                            
                            status.write("Creating index and ingesting data into OpenSearch...")
                            trainer.data_index.create()
                            trainer.data_index.bulk_addrecords(chunks_with_embeddings)

                            status.write("Setting up retriever...")
                            retriever = CompositeRetriever()
                            retriever.add_retriever("opensearch", OpenSearchRetriever(client, DATA_INDEX_NAME, embeddings))
                            st.session_state.retriever = retriever
                            
                            status.update(label="Training complete!", state="complete", expanded=False)

                    except Exception as e:
                        st.error(f"An error occurred during training: {e}", icon="🚨")
            else:
                st.warning("Please upload a file first.")

    st.header("Ask a Question")
    if st.session_state.retriever:
        query = st.text_input("Enter your query based on the trained document:")
        if query:
            with st.spinner("Retrieving results from OpenSearch..."):
                result_str, _ = st.session_state.retriever.get_documents(query=query)
                st.text_area("Retrieved Results", value=result_str, height=400)
    else:
        st.info("Please train on a document using the sidebar to enable querying.")
