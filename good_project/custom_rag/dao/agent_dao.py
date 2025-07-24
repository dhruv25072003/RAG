import logging

class AgentMetaIndex:
    """Mock AgentMetaIndex class."""
    def add_documents(self, agent_id, filenames):
        logging.info(f"DAO: add_documents() called for agent {agent_id} with files: {filenames}")
    def update_agent_details(self, agent_id, details):
        logging.info(f"DAO: update_agent_details() called for agent {agent_id} with details: {details}")

class AgentFileInfoIndex:
    """Mock AgentFileInfoIndex class."""
    def add_documents(self, agent_id, tenant_name, uploaded_by, docs, visual_cognition):
        logging.info(f"DAO: add_documents() called for agent {agent_id}.")
    def update_training_status(self, user_id, filenames, status):
        logging.info(f"DAO: update_training_status() called for user {user_id}, files: {filenames}, status: {status}")
    def add_images(self, uploaded_by, file_name, file_images):
        pass
    def get_doc_images(self, uploaded_by, doc):
        return []
    def delete_user_docs(self, uploaded_by, documents_to_remove):
        pass
