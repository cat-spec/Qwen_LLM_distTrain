from huggingface_hub import HfApi
from huggingface_hub import snapshot_download
from pathlib import Path
api = HfApi()
model_id="Qwen/Qwen3-0.6B-Base"
path_mode=Path(__file__).parent.parent/'src/disttrain/data'

local_dir = snapshot_download(
    repo_id="BelleGroup/train_0.5M_CN",
    repo_type="dataset",
    local_dir=path_mode,
    allow_patterns=["Belle_open_source_0.5M.json"],
)



