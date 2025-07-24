from langchain.schema import Document
import os

class ParentChildLangchainLoader:
    """Mocks the ParentChildLangchainLoader."""
    def __init__(self, *args, **kwargs):
        pass

    def load_and_convert(self, folder_path, shortlisted_files):
        docs = []
        for file_path in shortlisted_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            docs.append(Document(page_content=content, metadata={"source": os.path.basename(file_path)}))
        
        processed_files = shortlisted_files
        failed_files = []
        parent_lg_dict = {}
        all_docs_imgs_dict = {}
        return docs, processed_files, failed_files, parent_lg_dict, all_docs_imgs_dict
