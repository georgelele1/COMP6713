"""
Simple script to download and cache local model weights.
"""

from huggingface_hub import snapshot_download

# snapshot_download fetches all required files from the model repository
# into a local directory so that the inference scripts can later load them
# offline with local_files_only=True.
snapshot_download(
    repo_id="google/flan-t5-base",
    local_dir="T5/google_flan-t5-base",
)

snapshot_download(
    repo_id="nlptown/bert-base-multilingual-uncased-sentiment",
    local_dir="bert/nlptown_bert-base-multilingual-uncased-sentiment",
)
