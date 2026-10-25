"""Download only model/tokenizer data from a configuration's pinned revision."""
import argparse
import json
import re
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        raise ValueError("A full immutable model revision is required.")
    snapshot_download(
        config["model_id"], revision=config["revision"], max_workers=2,
        allow_patterns=["*.safetensors", "model.safetensors.index.json", "config.json",
                        "generation_config.json", "tokenizer*.json", "special_tokens_map.json",
                        "vocab.json", "merges.txt", "README.md"],
    )
    print(f"Cached {config['model_id']} at {config['revision']}; no remote code was executed.")


if __name__ == "__main__":
    main()
