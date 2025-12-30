"""
Purpose: Expand directory paths in manifests into newline-separated absolute file lists.
Arguments: None (reads paths from configs/paths.yaml)
Returns: Writes updated manifest files with one absolute path per line.
"""
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from scripts.utils.config_utils import load_config

def collect_images(root_dir: str):
    """Collect all image files from a directory recursively."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for p in Path(root_dir).rglob("*"):
        if p.suffix.lower() in exts:
            yield str(p.resolve())

def main():
    # Load config with variable substitution
    config_path = Path("configs/paths.yaml")
    
    if not config_path.exists():
        print("Config file not found, using default paths...")
        base_dir = Path(os.environ.get('FEATURE_EXTRACTOR_BASE',
                                       str(Path(__file__).parent.parent)))
        manifests = {
            "pp_manifest": base_dir / "data/manifests/pp_images.txt",
            "local_manifest": base_dir / "data/manifests/local_images.txt"
        }
        image_dirs = {
            "pp_manifest": base_dir / "data/place_pulse/images",
            "local_manifest": base_dir / "data/local_280/images"
        }
    else:
        paths = load_config(str(config_path))
        manifests = {
            "pp_manifest": Path(paths["data"]["pp_manifest"]),
            "local_manifest": Path(paths["data"]["local_manifest"])
        }
        image_dirs = {
            "pp_manifest": Path(paths["data"]["pp_images"]),
            "local_manifest": Path(paths["data"]["local_images"])
        }
    
    # Process each manifest
    for key, manifest_path in manifests.items():
        image_dir = image_dirs[key]
        
        # Check if manifest already exists and contains file list
        if manifest_path.exists():
            content = [l.strip() for l in open(manifest_path) if l.strip()]
            # If file contains exactly one line and it is a directory, expand it
            if len(content) == 1 and os.path.isdir(content[0]):
                print(f"Expanding directory path in {manifest_path}...")
                imgs = list(collect_images(content[0]))
            elif len(content) > 1:
                print(f"Manifest {manifest_path} already lists {len(content)} files, skipping...")
                continue
            else:
                # Empty or single non-directory entry, regenerate from image_dir
                print(f"Regenerating manifest {manifest_path} from {image_dir}...")
                imgs = list(collect_images(str(image_dir)))
        else:
            # Manifest doesn't exist, create from image_dir
            print(f"Creating new manifest {manifest_path} from {image_dir}...")
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            imgs = list(collect_images(str(image_dir)))
        
        # Sort images for consistency
        imgs.sort()
        
        # Write manifest
        manifest_path.write_text("\n".join(imgs) + "\n")
        print(f"Wrote {len(imgs)} entries to {manifest_path}")
        
        # Show first few entries for verification
        if imgs:
            print(f"  First entry: {Path(imgs[0]).name}")
            print(f"  Last entry: {Path(imgs[-1]).name}")

if __name__ == "__main__":
    main()