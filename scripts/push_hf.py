"""Upload a lerobot checkpoint or a lerobot v3.0 dataset to the HF Hub.

  python scripts/push_hf.py model   outputs/smolvla_libero_nvidia/checkpoints/last/pretrained_model USER/smolvla_libero_90 --readme MODEL_CARD.md
  python scripts/push_hf.py dataset outputs/libero90/nvidia/libero_90 USER/libero_90_lerobot_v3

Auth: `export HF_TOKEN=hf_...` (a write token) — the stored login is not touched.
Datasets get the `v3.0` tag LeRobotDataset resolves by default (moved to the new commit on re-upload); `--private` for a
private repo (readable only by org members).
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=["model", "dataset"])
    p.add_argument("src")
    p.add_argument("repo", help="namespace/name on the Hub")
    p.add_argument("--readme", help="markdown file uploaded as the repo README")
    p.add_argument("--private", action="store_true")
    p.add_argument("--message", default="upload")
    args = p.parse_args()

    src = Path(args.src).resolve()
    if args.kind == "model" and not (src / "model.safetensors").is_file():
        raise SystemExit(f"{src}: no model.safetensors (point at .../pretrained_model)")
    if args.kind == "dataset" and not (src / "meta/info.json").is_file():
        raise SystemExit(f"{src}: no meta/info.json (point at the dataset root)")

    api = HfApi()
    api.create_repo(args.repo, repo_type=args.kind, private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.repo, repo_type=args.kind, folder_path=str(src),
                      commit_message=args.message)
    if args.readme:
        api.upload_file(repo_id=args.repo, repo_type=args.kind, path_or_fileobj=args.readme,
                        path_in_repo="README.md", commit_message="readme")
    if args.kind == "dataset":
        # LeRobotDataset resolves revision "v3.0": the tag must point at THIS upload
        if "v3.0" in {t.name for t in api.list_repo_refs(args.repo, repo_type="dataset").tags}:
            api.delete_tag(args.repo, repo_type="dataset", tag="v3.0")
        api.create_tag(args.repo, repo_type="dataset", tag="v3.0")
    print(f"https://huggingface.co/{'datasets/' if args.kind == 'dataset' else ''}{args.repo}")


if __name__ == "__main__":
    main()
