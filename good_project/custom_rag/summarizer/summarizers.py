class ChunkSummarizer:
    """Mock ChunkSummarizer class."""
    def summarize(self, *args, **kwargs):
        return "This is a mock chunk summary."

class MapReduceSummarizer:
    """Mock MapReduceSummarizer class."""
    def summarize(self, *args, **kwargs):
        return {"output_text": "This is a mock map-reduce summary."}
