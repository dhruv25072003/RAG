import logging
from typing import Any, Dict, Tuple
from abc import ABC, abstractmethod
from enum import Enum

from opensearchpy import OpenSearch
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Abstract Base Class ---
class RetrieverType(str, Enum):
    Embeddings = 'embeddings'
    Documents = 'documents'
    Tabular = 'tabular'
    Tickets = 'tickets'

class Retriever(ABC):
    """
    Abstract base class for retriever.
    """
    @abstractmethod
    def get_documents(self,
                      query: str,
                      rerank: bool = False,
                      language: str = 'english',
                      keywords: str = '') -> Any:
        """
        get documents from the data store
        """
        pass

# --- Composite Retriever ---
# This class combines results from multiple retrievers.
class CompositeRetriever(Retriever):
    
    def __init__(self):
        super().__init__()
        self._retrievers: Dict[str, Retriever] = {}

    # This method correctly implements the abstract 'get_documents'
    def get_documents(self, *args, **kwargs) -> Tuple[str, Dict[str, Any]]:
        result_str = ''
        results = {}
        source_count_start = 1
        for name, retriever in self._retrievers.items():
            try:
                retriever_result_str, retriever_result_dict = retriever.get_documents(*args, **kwargs, source_count=source_count_start)
                result_str += f"\n\n--- Results from {name} ---\n{retriever_result_str}"
                results.update(retriever_result_dict)
                source_count_start = len(retriever_result_dict) + 1
            except Exception as e:
                logging.getLogger(CompositeRetriever.__name__).error(f"\n\nError retrieving documents with {name}: {e}")
        return result_str, results

    def retrieve_by_name(self, name: str, *args, **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Retrieve documents using a specific retriever by name.
        """
        if name not in self._retrievers:
            raise KeyError(f"No retriever found with the name '{name}'.")
        
        retriever = self._retrievers[name]
        try:
            retriever_result_str, retriever_result_dict = retriever.get_documents(*args, **kwargs)
            return retriever_result_str, retriever_result_dict
        except Exception as e:
            return f"Error retrieving documents with {name}: {e}", {}
        
    def add_retriever(self, name: str, retriever: Retriever) -> None:
        """Add a named retriever."""
        if name in self._retrievers:
            raise ValueError(f"A retriever with the name '{name}' already exists.")
        self._retrievers[name] = retriever

# --- Concrete OpenSearch Retriever ---
# This class was missing from the file, causing the ImportError.
class OpenSearchRetriever(Retriever):
    """A concrete retriever for fetching documents from OpenSearch."""

    def __init__(self, client: OpenSearch, index_name: str, embedding_model: HuggingFaceEmbeddings):
        self.client = client
        self.index_name = index_name
        self.embedding_model = embedding_model

    def get_documents(self, query: str, rerank: bool = False, language: str = 'english', keywords: str = '', **kwargs) -> Tuple[str, Dict[str, Any]]:
        """
        Get documents from the OpenSearch data store using k-NN search.
        """
        query_embedding = self.embedding_model.embed_query(query)

        search_query = {
            "size": 5,
            "query": {
                "knn": {
                    "embeddings": {
                        "vector": query_embedding,
                        "k": 5
                    }
                }
            }
        }

        response = self.client.search(
            index=self.index_name,
            body=search_query
        )

        result_str = "Retrieved Documents:\n\n"
        results_dict = {}
        source_count = kwargs.get('source_count', 1)

        for i, hit in enumerate(response['hits']['hits']):
            doc_text = hit['_source'].get('text', 'No text available.')
            doc_id = hit['_id']
            result_str += f"--- Source {source_count + i} (Score: {hit['_score']:.4f}) ---\n{doc_text}\n\n"
            results_dict[f"source_{source_count + i}"] = {
                "text": doc_text,
                "id": doc_id,
                "score": hit['_score']
            }

        if not response['hits']['hits']:
            result_str = "No relevant documents found in OpenSearch."

        return result_str, results_dict
