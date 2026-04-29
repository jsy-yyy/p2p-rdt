import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "google/embeddinggemma-300M"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download google/embeddinggemma-300M to a specified local path."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the model files will be saved.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face repo to download. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token. If omitted, uses local login / env settings.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=DEFAULT_HF_ENDPOINT,
        help=(
            "Hugging Face endpoint to use. "
            f"Default: {DEFAULT_HF_ENDPOINT}"
        ),
    )
    parser.add_argument(
        "--verify-load",
        action="store_true",
        help="Try loading the downloaded tokenizer and model from the local path after download.",
    )
    return parser.parse_args()


def verify_local_model(model_dir: Path) -> None:
    from sentence_transformers import SentenceTransformer

    print(f"Verifying local load from {model_dir} on CPU ...")
    model = SentenceTransformer(str(model_dir), device="cpu")
    dim = model.get_sentence_embedding_dimension()
    print(f"Local model load succeeded. sentence_embedding_dim={dim}")



def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

    print(f"Downloading {args.repo_id} to {output_dir} ...")
    if args.hf_endpoint:
        print(f"Using HF endpoint: {args.hf_endpoint}")
    downloaded_path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(output_dir),
        token=args.hf_token,
    )
    print(f"Download finished. Files saved under: {downloaded_path}")

    if args.verify_load:
        verify_local_model(output_dir)


if __name__ == "__main__":
    main()
