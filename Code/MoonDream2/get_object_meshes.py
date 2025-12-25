import os
import argparse
import trimesh
import objaverse
import fast_simplification as fs

def simplify_mesh(mesh, target_v=100000):
    if len(mesh.vertices) <= target_v:
        return mesh
    ratio = target_v / len(mesh.vertices)
    v, f = fs.simplify(mesh.vertices, mesh.faces, ratio)
    return trimesh.Trimesh(vertices=v, faces=f, process=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, default="labels.txt")
    parser.add_argument("--out", type=str, default="./meshes")
    args = parser.parse_args()

    if not os.path.exists(args.labels):
        return

    with open(args.labels, "r") as f:
        labels = [line.strip() for line in f if line.strip()]

    lvis = objaverse.load_lvis_annotations()
    os.makedirs(args.out, exist_ok=True)

    for label in labels:
        category = next((c for c in lvis.keys() if label in c or c in label), None)
        if not category: continue

        uids = lvis[category][:5]
        objects = objaverse.load_objects(uids=uids)
        
        for i, (uid, path) in enumerate(objects.items()):
            try:
                mesh = trimesh.load(path, force='mesh')
                mesh = simplify_mesh(mesh)
                mesh.export(os.path.join(args.out, f"{label}_{i}.obj"))
            except Exception as e:
                print(f"Error processing {uid}: {e}")

if __name__ == "__main__":
    main()
