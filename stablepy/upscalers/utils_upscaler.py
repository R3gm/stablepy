from __future__ import annotations
import os
from typing import Callable
from collections import namedtuple
import gc

import tqdm
import numpy as np
from PIL import Image
import spandrel
import torch
import torch.nn
import math

from stablepy.logging.logging_setup import logger

_spandrel_extra_init_state = None


def load_file_from_url(
    url: str,
    *,
    model_dir: str,
    progress: bool = True,
    file_name: str | None = None,
    re_download: bool = False,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.
    Returns the path to the downloaded file.

    file_name: if specified, it will be used as the filename, otherwise the filename will be extracted from the url.
        file is downloaded to {file_name}.tmp then moved to the final location after download is complete.
    re_download: forcibly re-download the file even if it already exists.
    """
    from urllib.parse import urlparse
    import requests

    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)

    cached_file = os.path.abspath(os.path.join(model_dir, file_name))

    if re_download or not os.path.exists(cached_file):
        os.makedirs(model_dir, exist_ok=True)
        temp_file = os.path.join(model_dir, f"{file_name}.tmp")
        logger.info(f'\nDownloading: "{url}" to {cached_file}')
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with tqdm.auto.tqdm(total=total_size, unit='B', unit_scale=True, desc=file_name, disable=not progress) as progress_bar:
            with open(temp_file, 'wb') as file:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))

        os.rename(temp_file, cached_file)
    return cached_file


def _init_spandrel_extra_archs() -> None:
    """
    Try to initialize `spandrel_extra_archs` (exactly once).
    """
    global _spandrel_extra_init_state
    if _spandrel_extra_init_state is not None:
        return

    try:
        import spandrel
        import spandrel_extra_arches
        spandrel.MAIN_REGISTRY.add(*spandrel_extra_arches.EXTRA_REGISTRY)
        _spandrel_extra_init_state = True
    except Exception:
        logger.warning("Failed to load spandrel_extra_arches", exc_info=True)
        _spandrel_extra_init_state = False


def load_spandrel_model(
    path: str | os.PathLike,
    *,
    device: str | torch.device | None,
    prefer_half: bool = False,
    dtype: str | torch.dtype | None = None,
    expected_architecture: str | None = None,
) -> spandrel.ModelDescriptor:
    global _spandrel_extra_init_state

    import spandrel
    _init_spandrel_extra_archs()

    model_descriptor = spandrel.ModelLoader(device=device).load_from_file(str(path))
    arch = model_descriptor.architecture
    if expected_architecture and arch.name != expected_architecture:
        logger.warning(
            f"Model {path!r} is not a {expected_architecture!r} model (got {arch.name!r})",
        )
    half = False
    if prefer_half:
        if model_descriptor.supports_half:
            model_descriptor.model.half()
            half = True
        else:
            logger.info("Model %s does not support half precision, ignoring half", path)
    if dtype:
        model_descriptor.model.to(dtype=dtype)
    model_descriptor.model.eval()
    logger.debug(
        "Loaded %s from %s (device=%s, half=%s, dtype=%s)",
        arch, path, device, half, dtype,
    )
    return model_descriptor


def get_param(model) -> torch.nn.Parameter:
    """
    Find the first parameter in a model or module.
    """
    if hasattr(model, "model") and hasattr(model.model, "parameters"):
        # Unpeel a model descriptor to get at the actual Torch module.
        model = model.model

    for param in model.parameters():
        return param

    raise ValueError(f"No parameters found in model {model!r}")


class Grid(namedtuple("_Grid", ["tiles", "tile_w", "tile_h", "image_w", "image_h", "overlap"])):
    @property
    def tile_count(self) -> int:
        """
        The total number of tiles in the grid.
        """
        return sum(len(row[2]) for row in self.tiles)


def split_grid(image: Image.Image, tile_w: int = 512, tile_h: int = 512, overlap: int = 64) -> Grid:
    w, h = image.size

    non_overlap_width = tile_w - overlap
    non_overlap_height = tile_h - overlap

    cols = math.ceil((w - overlap) / non_overlap_width)
    rows = math.ceil((h - overlap) / non_overlap_height)

    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    grid = Grid([], tile_w, tile_h, w, h, overlap)
    for row in range(rows):
        row_images = []

        y = int(row * dy)

        if y + tile_h >= h:
            y = h - tile_h

        for col in range(cols):
            x = int(col * dx)

            if x + tile_w >= w:
                x = w - tile_w

            tile = image.crop((x, y, x + tile_w, y + tile_h))

            row_images.append([x, tile_w, tile])

        grid.tiles.append([y, tile_h, row_images])

    return grid


def combine_grid(grid):
    def make_mask_image(r):
        r = r * 255 / grid.overlap
        r = r.astype(np.uint8)
        return Image.fromarray(r, 'L')

    mask_w = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((1, grid.overlap)).repeat(grid.tile_h, axis=0))
    mask_h = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((grid.overlap, 1)).repeat(grid.image_w, axis=1))

    combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
    for y, h, row in grid.tiles:
        combined_row = Image.new("RGB", (grid.image_w, h))
        for x, w, tile in row:
            if x == 0:
                combined_row.paste(tile, (0, 0))
                continue

            combined_row.paste(tile.crop((0, 0, grid.overlap, h)), (x, 0), mask=mask_w)
            combined_row.paste(tile.crop((grid.overlap, 0, w, h)), (x + grid.overlap, 0))

        if y == 0:
            combined_image.paste(combined_row, (0, 0))
            continue

        combined_image.paste(combined_row.crop((0, 0, combined_row.width, grid.overlap)), (0, y), mask=mask_h)
        combined_image.paste(combined_row.crop((0, grid.overlap, combined_row.width, h)), (0, y + grid.overlap))

    return combined_image


def pil_image_to_torch_bgr(img: Image.Image) -> torch.Tensor:
    img = np.array(img.convert("RGB"))
    img = img[:, :, ::-1]  # flip RGB to BGR
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img) / 255  # Rescale to [0, 1]
    return torch.from_numpy(img)


def torch_bgr_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    if tensor.ndim == 4:
        # If we're given a tensor with a batch dimension, squeeze it out
        # (but only if it's a batch of size 1).
        if tensor.shape[0] != 1:
            raise ValueError(f"{tensor.shape} does not describe a BCHW tensor")
        tensor = tensor.squeeze(0)
    assert tensor.ndim == 3, f"{tensor.shape} does not describe a CHW tensor"
    # TODO: is `tensor.float().cpu()...numpy()` the most efficient idiom?
    arr = tensor.float().cpu().clamp_(0, 1).numpy()  # clamp
    arr = 255.0 * np.moveaxis(arr, 0, 2)  # CHW to HWC, rescale
    arr = arr.round().astype(np.uint8)
    arr = arr[:, :, ::-1]  # flip BGR to RGB
    return Image.fromarray(arr, "RGB")


def upscale_pil_patch(model, img: Image.Image) -> Image.Image:
    """
    Upscale a given PIL image using the given model.
    """
    param = get_param(model)

    with torch.inference_mode():
        tensor = pil_image_to_torch_bgr(img).unsqueeze(0)  # add batch dimension
        tensor = tensor.to(device=param.device, dtype=param.dtype)
        return torch_bgr_to_pil_image(model(tensor))


def upscale_with_model(
    model: Callable[[torch.Tensor], torch.Tensor],
    img: Image.Image,
    *,
    tile_size: int,
    tile_overlap: int = 0,
    desc="tiled upscale",
    disable_progress_bar=False,
) -> Image.Image:
    if tile_size <= 0:
        logger.debug("Upscaling %s without tiling", img)
        output = upscale_pil_patch(model, img)
        logger.debug("=> %s", output)
        return output

    grid = split_grid(img, tile_size, tile_size, tile_overlap)
    newtiles = []

    with tqdm.auto.tqdm(total=grid.tile_count, desc=desc, disable=disable_progress_bar) as p:
        for y, h, row in grid.tiles:
            newrow = []
            for x, w, tile in row:
                output = upscale_pil_patch(model, tile)
                scale_factor = output.width // tile.width
                newrow.append([x * scale_factor, w * scale_factor, output])
                p.update(1)
            newtiles.append([y * scale_factor, h * scale_factor, newrow])

    newgrid = Grid(
        newtiles,
        tile_w=grid.tile_w * scale_factor,
        tile_h=grid.tile_h * scale_factor,
        image_w=grid.image_w * scale_factor,
        image_h=grid.image_h * scale_factor,
        overlap=grid.overlap * scale_factor,
    )
    return combine_grid(newgrid)


def tiled_upscale_2(
    img: torch.Tensor,
    model,
    *,
    tile_size: int,
    tile_overlap: int,
    scale: int,
    device: torch.device,
    desc="Tiled upscale",
    disable_progress_bar=False,
):
    # Alternative implementation of `upscale_with_model` originally used by
    # SwinIR and ScuNET.  It differs from `upscale_with_model` in that tiling and
    # weighting is done in PyTorch space, as opposed to `images.Grid` doing it in
    # Pillow space without weighting.

    b, c, h, w = img.size()
    tile_size = min(tile_size, h, w)

    if tile_size <= 0:
        logger.debug("Upscaling %s without tiling", img.shape)
        return model(img)

    stride = tile_size - tile_overlap
    h_idx_list = list(range(0, h - tile_size, stride)) + [h - tile_size]
    w_idx_list = list(range(0, w - tile_size, stride)) + [w - tile_size]
    result = torch.zeros(
        b,
        c,
        h * scale,
        w * scale,
        device=device,
        dtype=img.dtype,
    )
    weights = torch.zeros_like(result)
    logger.debug("Upscaling %s to %s with tiles", img.shape, result.shape)
    with tqdm.auto.tqdm(total=len(h_idx_list) * len(w_idx_list), desc=desc, disable=disable_progress_bar) as pbar:
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:

                # Only move this patch to the device if it's not already there.
                in_patch = img[
                    ...,
                    h_idx: h_idx + tile_size,
                    w_idx: w_idx + tile_size,
                ].to(device=device)

                out_patch = model(in_patch)

                result[
                    ...,
                    h_idx * scale: (h_idx + tile_size) * scale,
                    w_idx * scale: (w_idx + tile_size) * scale,
                ].add_(out_patch)

                out_patch_mask = torch.ones_like(out_patch)

                weights[
                    ...,
                    h_idx * scale: (h_idx + tile_size) * scale,
                    w_idx * scale: (w_idx + tile_size) * scale,
                ].add_(out_patch_mask)

                pbar.update(1)

    output = result.div_(weights)

    return output


def upscale_2(
    img: Image.Image,
    model,
    *,
    tile_size: int,
    tile_overlap: int,
    scale: int,
    desc: str,
    disable_progress_bar: bool,
):
    """
    Convenience wrapper around `tiled_upscale_2` that handles PIL images.
    """
    param = get_param(model)
    tensor = pil_image_to_torch_bgr(img).to(device=model.device, dtype=param.dtype).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        output = tiled_upscale_2(
            tensor,
            model,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            scale=scale,
            desc=desc,
            device=param.device,
            disable_progress_bar=disable_progress_bar,
        )
    return torch_bgr_to_pil_image(output)


def release_resources_upscaler():
    torch.cuda.empty_cache()
    gc.collect()
