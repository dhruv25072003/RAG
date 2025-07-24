import os

class KnowledgeSettings:
    """Mock KnowledgeSettings class."""
    def __init__(self, temp_dir):
        self.agent_input_path = temp_dir
        self.public_data_path = os.path.join(temp_dir, "public")
        self.image_data_path = os.path.join(temp_dir, "images")
        os.makedirs(self.public_data_path, exist_ok=True)
        os.makedirs(self.image_data_path, exist_ok=True)
