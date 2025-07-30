import streamlit as st
import os
import tempfile
from opensearchpy import OpenSearch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from typing import Any, Dict, Tuple, List

# --- Correctly import from the custom_rag package ---
from custom_rag.retriever import CompositeRetriever, OpenSearchRetriever
from custom_rag.loaders.langchain_loader import ParentChildLangchainLoader

# --- App Configuration ---
st.set_page_config(page_title="Unified RAG Pipeline", layout="wide")
st.title("RAG Pipeline: Query Files, Databases, Snowflake, and PostgreSQL")

# --- Global Settings ---
OPENSEARCH_URL = "http://localhost:9200"
DATA_INDEX_NAME = "my-unified-rag-index"

# --- Concrete OpenSearch Index Class ---
class OpenSearchIndex:
    def __init__(self, client, index_name):
        self.client = client
        self.index_name = index_name

    def create(self):
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
        index_body = {
            "settings": {"index": {"knn": True}},
            "mappings": {"properties": {"embeddings": {"type": "knn_vector", "dimension": 384}}}
        }
        self.client.indices.create(index=self.index_name, body=index_body)

    def bulk_addrecords(self, records: List[Document]):
        for doc in records:
            self.client.index(
                index=self.index_name,
                body={"text": doc.page_content, "embeddings": doc.metadata['embeddings'], "source": doc.metadata.get('source')},
                refresh=True
            )

# --- Main App Logic ---
@st.cache_resource
def get_global_resources():
    try:
        client = OpenSearch(OPENSEARCH_URL, timeout=30)
        if not client.ping(): raise ConnectionError("Could not connect to OpenSearch.")
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        loader = ParentChildLangchainLoader()
        
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        llm_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
        llm = HuggingFacePipeline(pipeline=llm_pipeline)

        return client, embeddings, loader, llm
    except Exception as e:
        st.error(f"Failed to initialize resources: {e}")
        return None, None, None, None

def process_and_index_documents(docs: List[Document]):
    if not docs:
        st.warning("No documents were loaded to process.")
        return

    st.session_state.retriever = None
    with st.status("Processing documents...", expanded=True) as status:
        status.write("Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(docs)
        status.write(f"Split into {len(chunks)} chunks.")

        status.write("Generating embeddings...")
        sentences = [chunk.page_content for chunk in chunks]
        embeddings_list = embeddings.embed_documents(sentences)
        for chunk, embed in zip(chunks, embeddings_list):
            chunk.metadata['embeddings'] = embed
        
        status.write("Creating index and ingesting data into OpenSearch...")
        data_index = OpenSearchIndex(client, DATA_INDEX_NAME)
        data_index.create()
        data_index.bulk_addrecords(chunks)

        status.write("Setting up retriever...")
        retriever = CompositeRetriever()
        retriever.add_retriever("opensearch", OpenSearchRetriever(client, DATA_INDEX_NAME, embeddings))
        st.session_state.retriever = retriever
        
        status.update(label="Training complete!", state="complete", expanded=False)

# --- Streamlit UI ---
client, embeddings, loader, llm = get_global_resources()

if 'retriever' not in st.session_state:
    st.session_state.retriever = None

if not client or not llm:
    st.error("Application cannot start. Please ensure OpenSearch is running and check your internet connection for model downloads.")
else:
    st.sidebar.title("Data Sources")
    source_option = st.sidebar.radio("Choose a data source:", ("File Upload", "Snowflake", "PostgreSQL"))

    # --- File Upload Section ---
    if source_option == "File Upload":
        st.sidebar.header("File Upload")
        uploaded_file = st.sidebar.file_uploader(
            "Upload a file",
            type=["txt", "md", "csv", "db", "sqlite", "sqlite3"]
        )
        if st.sidebar.button("Train from File"):
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_file_path = tmp_file.name
                
                docs = loader.load_from_file(tmp_file_path)
                os.remove(tmp_file_path)
                
                process_and_index_documents(docs)
            else:
                st.sidebar.warning("Please upload a file first.")

    # --- Snowflake Section ---
    elif source_option == "Snowflake":
        st.sidebar.header("Snowflake Connection")
        with st.sidebar.form("snowflake_form"):
            sf_user = st.text_input("User")
            sf_password = st.text_input("Password", type="password")
            sf_account = st.text_input("Account")
            sf_database = st.text_input("Database")
            sf_schema = st.text_input("Schema")
            sf_table = st.text_input("Table")
            submitted = st.form_submit_button("Train from Snowflake")
            if submitted:
                if all([sf_user, sf_password, sf_account, sf_database, sf_schema, sf_table]):
                    try:
                        docs = loader.load_from_snowflake(sf_user, sf_password, sf_account, sf_database, sf_schema, sf_table)
                        process_and_index_documents(docs)
                    except Exception as e:
                        st.sidebar.error(f"Snowflake Error: {e}")
                else:
                    st.sidebar.warning("Please fill in all Snowflake details.")

    # --- PostgreSQL Section ---
    elif source_option == "PostgreSQL":
        st.sidebar.header("PostgreSQL Connection")
        with st.sidebar.form("postgres_form"):
            pg_user = st.text_input("User")
            pg_password = st.text_input("Password", type="password")
            pg_host = st.text_input("Host", "localhost")
            pg_port = st.text_input("Port", "5432")
            pg_dbname = st.text_input("Database Name")
            pg_table = st.text_input("Table Name")
            submitted = st.form_submit_button("Train from PostgreSQL")
            if submitted:
                if all([pg_user, pg_password, pg_host, pg_port, pg_dbname, pg_table]):
                    try:
                        docs = loader.load_from_postgres(pg_user, pg_password, pg_host, pg_port, pg_dbname, pg_table)
                        process_and_index_documents(docs)
                    except Exception as e:
                        st.sidebar.error(f"PostgreSQL Error: {e}")
                else:
                    st.sidebar.warning("Please fill in all PostgreSQL details.")

    # --- Main Q&A Area ---
    st.header("Ask a Question")
    if st.session_state.retriever:
        query = st.text_input("Enter your query to retrieve and summarize information:")
        if query:
            with st.spinner("Retrieving and summarizing..."):
                retrieved_text, _ = st.session_state.retriever.get_documents(query=query)
                prompt = f"Based on the following context, please provide a concise answer to the user's question.\n\nContext:\n{retrieved_text}\n\nQuestion:\n{query}\n\nAnswer:"
                answer = llm(prompt)
                
                st.subheader("Summarized Answer")
                # FIX: The 'answer' from the pipeline is a direct string, not a list of dicts.
                st.write(answer)

                with st.expander("Show Retrieved Context"):
                    st.text_area("", value=retrieved_text, height=300)
    else:
        st.info("Please train on a data source using the sidebar to enable querying.")
