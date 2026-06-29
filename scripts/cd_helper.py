import torch
import numpy as np
import trimesh
from argparse import ArgumentParser
import os
import json
from chamfer_distance import ChamferDistance

def _as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        assert len(scene_or_mesh.geometry) > 0
        mesh = trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                for g in scene_or_mesh.geometry.values())
        )
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh

def _sample_mesh(m, n):
    vpos, _ = trimesh.sample.sample_surface(m, n)
    return torch.tensor(vpos, dtype=torch.float32, device="cuda")

def compute_distance(compared_path: str, reference_path: str, sample_point_count: int):
    chamfer_dist = ChamferDistance()
    ref = _as_mesh(trimesh.load(reference_path))
    mesh = _as_mesh(trimesh.load(compared_path))
    vpos_mesh = _sample_mesh(mesh, sample_point_count)
    vpos_ref = _sample_mesh(ref, sample_point_count)
    dist1, dist2, idx1, idx2 = chamfer_dist(vpos_mesh[None, ...], vpos_ref[None, ...])
    loss = (torch.mean(dist1) + torch.mean(dist2)).item()
    return mesh.faces.shape[0], ref.faces.shape[0], loss

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-n', type=int, default=100000)
    parser.add_argument('comp_path', type=str)
    parser.add_argument('ref_path', type=str)
    parser.add_argument('cd_json_output_dir', type=str)
    args = parser.parse_args()
    tris_compared, tris_reference, loss = compute_distance(args.comp_path, args.ref_path, args.n)
    print(f"Tris of compared: {tris_compared}; Tris of reference: {tris_reference}; Chamfer loss: {loss}")
    # Save the results to a JSON file
    output_path = os.path.join(args.cd_json_output_dir, "cd_results.json")
    data = {
        "tris_compared": tris_compared,
        "tris_reference": tris_reference,
        "loss": loss
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Results saved to: {output_path}")