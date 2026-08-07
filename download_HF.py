from huggingface_hub import snapshot_download
import os

# Temporarily unset proxies in the Python process
for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "HF_PROXY"]:
    os.environ.pop(var, None)

snapshot_download(
    repo_id="nvidia/nemotron-3.5-asr-streaming-0.6b",
    local_dir="./nemotron-3.5-asr-streaming-0.6b"
)