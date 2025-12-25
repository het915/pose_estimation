import os
import argparse
import trimesh
import objaverse.xl as oxl

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, default="labels.txt")
    parser.add_argument("--out", type=str, default="./meshes_xl")
    args = parser.parse_args()

    with open(args.labels, "r") as f:
        labels = [line.strip() for line in f if line.strip()]

    annotations = oxl.get_annotations()
    os.makedirs(args.out, exist_ok=True)

    for label in labels:
        matches = annotations[annotations['metadata'].str.contains(label, case=False, na=False)]
        if matches.empty: continue

        downloaded = oxl.download_objects(matches.head(5))
        for i, (file_id, path) in enumerate(downloaded.items()):
            try:
                mesh = trimesh.load(path, force='mesh')
                mesh.export(os.path.join(args.out, f"{label}_xl_{i}.obj"))
            except Exception as e:
                print(f"Failed XL conversion for {file_id}: {e}")

if __name__ == "__main__":
    main()
