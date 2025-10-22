import os
import argparse
import datetime
from typing import Union, Optional

import torch
from PIL import Image
import trimesh

from direct3d_s2.pipeline import Direct3DS2Pipeline


def load_pipeline_512(device: str = "cuda:0") -> Direct3DS2Pipeline:
    """Load pipeline and move ONLY 512-path modules to the target device.

    This avoids moving 1024 models to GPU memory when generating at 512.
    """
    pipe = Direct3DS2Pipeline.from_pretrained(
        'wushuang98/Direct3D-S2',
        subfolder="direct3d-s2-v-1-1",
    )

    # Manually set device for internal use (e.g., background remover)
    pipe.device = torch.device(device)

    # Move only required components to the device for 512 generation
    pipe.dense_vae.to(device)
    pipe.dense_dit.to(device)
    pipe.sparse_vae_512.to(device)
    pipe.sparse_dit_512.to(device)
    pipe.refiner.to(device)
    pipe.dense_image_encoder.to(device)
    pipe.sparse_image_encoder.to(device)

    # Keep 1024-path modules on CPU to save VRAM
    # pipe.sparse_vae_1024, pipe.sparse_dit_1024, pipe.refiner_1024 remain on CPU

    return pipe


def image_to_mesh_512(
    image: Union[str, Image.Image],
    output_dir: str = "outputs/mesh_512",
    simplify: bool = True,
    simplify_ratio: float = 0.95,
    mc_threshold: float = 0.2,
    refine_method: str = "batch",
    device: str = "cuda:0",
) -> str:
    """Run image-to-mesh at 512 resolution only and save OBJ.

    Returns the saved OBJ path.
    """
    os.makedirs(output_dir, exist_ok=True)

    pipe = load_pipeline_512(device=device)

    uid = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    png_path = os.path.join(output_dir, f"{uid}.png")
    obj_path = os.path.join(output_dir, f"{uid}.obj")

    if isinstance(image, str):
        img = Image.open(image)
    else:
        img = image
    img.save(png_path)

    outputs = pipe(
        img,
        sdf_resolution=512,  # force 512 path
        remove_interior=True,
        remesh=simplify,
        simplify_ratio=simplify_ratio,
        mc_threshold=mc_threshold,
        refine_method=refine_method,
    )

    mesh: trimesh.Trimesh = outputs["mesh"]
    mesh.export(obj_path, include_normals=True)

    return obj_path


def main():
    parser = argparse.ArgumentParser(description="Image to Mesh at 512 resolution only")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--output_dir", type=str, default="outputs/mesh_512", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device, e.g., cuda:0 or cpu")
    parser.add_argument("--simplify", action="store_true", help="Enable mesh simplification")
    parser.add_argument("--simplify_ratio", type=float, default=0.95, help="Faces reduction ratio if simplify enabled")
    parser.add_argument("--mc_threshold", type=float, default=0.2, help="Marching cubes threshold")
    parser.add_argument("--refine_method", type=str, default="batch", choices=["loop","batch"], help="Refine method: loop or batch")

    args = parser.parse_args()

    obj_path = image_to_mesh_512(
        image=args.input,
        output_dir=args.output_dir,
        simplify=args.simplify,
        simplify_ratio=args.simplify_ratio,
        mc_threshold=args.mc_threshold,
        refine_method=args.refine_method,
        device=args.device,
    )

    print(f"Saved: {obj_path}")


if __name__ == "__main__":
    main()


