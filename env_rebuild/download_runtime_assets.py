"""Fetch only the small SmolVLM config and processor files needed at inference."""
from pathlib import Path
from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent
DEST = HERE / "inputs/smolvlm2_processor"
FILES = ["config.json", "processor_config.json", "preprocessor_config.json",
         "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
         "added_tokens.json", "merges.txt", "vocab.json", "chat_template.json"]
path = snapshot_download("HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                         local_dir=DEST, allow_patterns=FILES)
print(path)
