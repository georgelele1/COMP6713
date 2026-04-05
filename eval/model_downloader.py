"""下载并缓存本地模型权重的简单脚本。

当前脚本用于把 Hugging Face 上的 `google/flan-t5-base`
下载到项目本地的 T5/google_flan-t5-base 目录中。

如果后续你想切换到其他模型，也可以仿照这里修改 repo_id 与 local_dir。
"""

from huggingface_hub import snapshot_download

# snapshot_download 会把模型仓库中的全部必要文件下载到本地目录，
# 后续推理脚本即可通过 local_files_only=True 直接离线加载。
snapshot_download(
    repo_id="google/flan-t5-base",  # nlptown/bert-base-multilingual-uncased-sentiment
    local_dir="T5/google_flan-t5-base",  # bert/nlptown_bert-base-multilingual-uncased-sentiment
)
