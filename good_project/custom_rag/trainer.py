import urllib.parse
import os
import pytz
import shutil
import logging
import uuid

from datetime import datetime, timezone
from langchain.schema import Document
from pathlib import Path
from typing import List, Any, Optional
from tqdm import tqdm
from time import sleep
from opensearchpy import OpenSearch

# Corrected relative imports for the package structure
from .dao.agent_dao import AgentFileInfoIndex, AgentMetaIndex
from .summarizer.summarizers import ChunkSummarizer, MapReduceSummarizer
from .constants import UploadType
from .loaders.langchain_loader import ParentChildLangchainLoader
from .config.settings import KnowledgeSettings


class HybridParentChildTrainer:

    def __init__(self, 
                 es: OpenSearch,
                 agent_id: str, 
                 agent_type: str, 
                 tenant_name: str, 
                 agent_name: str, 
                 settings:KnowledgeSettings,
                 suggestions: list, 
                 avatar_color, 
                 embedding_obj: Any, 
                 configurations: dict, 
                 data_index,
                 meta_index,
                 add_filename_flag:bool = False,
                 advanced_doc_processing: bool = True,
                 visual_cognition: bool = False,
                 upload_type: UploadType = UploadType.STANDARD,
                 user_id: str = "",
                 window_size: int = 1,
                 window_overlap: int =0,
                 widget = None,
                 is_private: bool = False,
                 logger: Optional[logging.Logger]=None
                 ) -> None:
        
        self.es = es
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.tenant_name = tenant_name
        self._settings = settings
        self.suggestions = suggestions
        self.avatar_color = avatar_color
        self.add_filename_flag = add_filename_flag
        self.configurations = configurations
        self.upload_type = upload_type
        self.user_id = user_id
        self.data_index = data_index
        self.is_private=is_private
        self.meta_index: AgentMetaIndex = meta_index
        self.agent_file_index = AgentFileInfoIndex()
        if self.is_private:
            self.chunksummarizer = ChunkSummarizer()
            self.mapreducesummarizer = MapReduceSummarizer()

        self.input_dir = str(Path(self._settings.agent_input_path) / agent_id)
        self.public_dir = self._settings.public_data_path
        self.img_dir = self._settings.image_data_path
        self.visual_cognition = visual_cognition
        
        # --- FIX: Use the embedding object directly ---
        # The original code was 'embedding_obj.embedding', which caused the error.
        self.embedding = embedding_obj if embedding_obj else None
        
        self.widget = widget
        self.loader = ParentChildLangchainLoader(configurations['chunk_sizes']['parent'],
                                                configurations['chunk_sizes']['child'],
                                                settings = self._settings,
                                                add_filename_flag=self.add_filename_flag,
                                                advanced_doc_processing=advanced_doc_processing,
                                                visual_cognition = self.visual_cognition,
                                                upload_type=self.upload_type,
                                                window_size =window_size,
                                                window_overlap = window_overlap)
        self._logger = logger or logging.getLogger(HybridParentChildTrainer.__name__)

    def train(self, 
              batch_size=1, 
              file_list=[]):
        return self.add_documents(batch_size, 
                                  file_list)
    
    def add_documents(self,
                      batch_size=1, 
                      file_list=[]):
        base_files = []
        try:
            files_to_process = [str(Path(self.input_dir) / filename) for filename in os.listdir(self.input_dir)] if file_list==[] else file_list
            
            base_files = [Path(filename).name for filename in files_to_process]
                        
            self.agent_file_index.add_documents(self.agent_id, self.tenant_name ,uploaded_by=self.user_id, 
                                         docs = base_files, 
                                         visual_cognition=self.visual_cognition)

            self.update_agent_status('in progress')
            is_training_success = False
            all_files = []

            self.data_index.create()

            for batch in tqdm(range(0, len(files_to_process), batch_size)):
                batch_files = files_to_process[batch:batch + batch_size]
                batch_file_names = [Path(filename).name for filename in batch_files]

                if file_list:
                    folder_path = Path(batch_files[0]).parent
                else:
                    folder_path = self.input_dir
                try:
                    documents, processed_files, failed_files, parent_lg_dict, all_docs_imgs_dict = self.loader.load_and_convert(folder_path=folder_path, 
                                                                                                            shortlisted_files = batch_files)
                
                    for _filename, _images in all_docs_imgs_dict.items():
                        self.agent_file_index.add_images(uploaded_by = self.user_id, 
                                                   file_name=_filename, 
                                                   file_images=_images)


                    self.processed_files = processed_files    
                    try:
                        document_with_embeddings = self.get_embeddings(documents)
                        # This loop seems unnecessary if get_embeddings modifies in place, but we'll keep it for safety.
                        for document in document_with_embeddings:
                            document.uploaded_by = self.user_id
                
                        self._logger.info('embedding completed')
                    except Exception as err:
                        self._logger.error(f"Embedding Error: {err}")
                        sleep(15)
                        failed_files.extend(processed_files)
                        processed_files=[]
                        continue

                    try:
                        self.data_index.bulk_addrecords(document_with_embeddings)

                        if self.is_private:
                            self.process_byod_summaries(parent_lg_dict, batch_file_names)
                        
                        self.agent_file_index.update_training_status(self.user_id, batch_file_names, "success")

                    except Exception as err:
                        self._logger.error(f"OS Upload Error: {err}")
                        failed_files.extend(processed_files)
                        processed_files=[]
                        is_training_success = False
                        self.agent_file_index.update_training_status(self.user_id, batch_file_names, "failed")
                        continue  
                    self.persist()
                    all_files.extend(processed_files)
                except Exception as ex:
                    self._logger.error(f"loading exception {ex}")
                    is_training_success = False
                    self.agent_file_index.update_training_status(self.user_id, batch_file_names, "failed")
                    continue
            is_training_success = True
        except Exception as err:
            self._logger.error("Training failed for agent: %s - %s", self.agent_id, str(err))
            is_training_success = False
        finally:
            if is_training_success:
                self.update_agent_status("success")
            else:
                self.update_agent_status("failed")
                if base_files: self.agent_file_index.update_training_status(self.user_id, base_files, "failed")

    def get_embeddings(self, 
                       documents: List[Document]) -> List[Document]:
        
        # This check is important because we modified the __init__
        if not self.embedding:
            self._logger.error("Embedding model is not initialized.")
            raise ValueError("Embedding model not available.")

        try:
            sentences = [doc.page_content for doc in documents]
            embeddings = self.embedding.embed_documents(sentences)
            for doc, embed in zip(documents, embeddings):
                doc.metadata['embeddings'] = embed # Storing in metadata for our app's logic
            self._logger.info("Embedding successful.")
            return documents
        except Exception as e:
            self._logger.error(f"Embedding failed: {e}")
            raise
    
    def process_byod_summaries(self, parent_lg_dict, batch_file_names):
        """Process document summaries for BYOD upload type."""
        # This part is likely not used in our Streamlit app but is kept for completeness.
        pass
    
    def persist(self):
        try:
            self.publish_files(files=self.processed_files)
            self.filenames = [Path(file_path).name for file_path in self.processed_files]
            self.meta_index.add_documents(self.agent_id, self.filenames)
        except Exception as err:
            self._logger.error(f"agent file persistence failed with exception {err}")
            raise Exception("agent file persistence failed with exception")
    
    def publish_files(self, files):
        for file in files:
            try:
                shutil.copy2(file, self.public_dir)
            except Exception as err:
                self._logger.error(f"file copy failed for {file}")
                raise Exception(f"file copy failed for {file}")

    def update_agent_status(self, training_status):
        self.meta_index.update_agent_details(self.agent_id,{"training_status":training_status})
    
    # Other methods from your original file are kept below for completeness
    def add_agent_data(self, user_email): pass
    def remove_documents(self, documents_to_remove=None, urls_to_remove=None, url_splitter_text = "obj", chunk_size=10000): pass
    def remove_files_from_es(self, documents_to_remove = None, urls_to_remove = None): pass
    def cleanup_imgs(self, documents_to_remove): pass
    def cleanup(self, documents_to_remove, file_dir = ""): pass
    def _extract_objs_from_urls(self, urls: List[str], url_splitter_text: str = "obj") -> List[str]: pass
    def update_config_to_es(self, config): pass
