# =====================================
# BASE UPSCALER
# =====================================
import torch
device_upscaler = "cuda" if torch.cuda.is_available() else "cpu"  # run in CPU is very slow

from abc import abstractmethod
import PIL
from PIL import Image

LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
NEAREST = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST


class Upscaler:
    name = None
    model_path = None
    model_name = None
    model_url = None
    enable = True
    filter = None
    model = None
    user_path = None
    tile = True

    def __init__(self, create_dirs=False):
        self.mod_pad_h = None
        self.tile_size = 100
        self.tile_pad = 10
        self.device = device_upscaler
        self.img = None
        self.output = None
        self.scale = 4.0
        self.half = True
        self.pre_pad = 0
        self.mod_scale = None
        self.model_download_path = "upscalers"

        if self.model_path is None and self.name:
            self.model_path = os.path.join(
                "/content/Real-ESRGAN/weights/RealESRGAN_x2plus.pth", self.name
            )
        if self.model_path and create_dirs:
            os.makedirs(self.model_path, exist_ok=True)

        try:
            import cv2  # noqa: F401

            self.can_tile = True
        except Exception:
            pass

    @abstractmethod
    def do_upscale(self, img: PIL.Image, selected_model: str):
        return img

    def upscale(self, img: PIL.Image, scale, selected_model: str = None):
        self.scale = scale
        dest_w = int((img.width * scale) // 8 * 8)
        dest_h = int((img.height * scale) // 8 * 8)

        for _ in range(3):
            shape = (img.width, img.height)

            img = self.do_upscale(img, selected_model)

            if shape == (img.width, img.height):
                break

            if img.width >= dest_w and img.height >= dest_h:
                break

        if img.width != dest_w or img.height != dest_h:
            img = img.resize((int(dest_w), int(dest_h)), resample=LANCZOS)

        return img

    @abstractmethod
    def load_model(self, path: str):
        pass

    def update_status(self, prompt):
        print(f"extras: {prompt}")


class UpscalerNone(Upscaler):
    name = "None"

    def load_model(self, path):
        pass

    def do_upscale(self, img, selected_model=None):
        return img

    def __init__(self, dirname=None):
        super().__init__(False)


class UpscalerLanczos(Upscaler):

    def do_upscale(self, img, selected_model=None):
        return img.resize(
            (int(img.width * self.scale), int(img.height * self.scale)),
            resample=LANCZOS,
        )

    def load_model(self, _):
        pass

    def __init__(self, dirname=None):
        super().__init__(False)
        self.name = "Lanczos"


class UpscalerNearest(Upscaler):

    def do_upscale(self, img, selected_model=None):
        return img.resize(
            (int(img.width * self.scale), int(img.height * self.scale)),
            resample=NEAREST,
        )

    def load_model(self, _):
        pass

    def __init__(self, dirname=None):
        super().__init__(False)
        self.name = "Nearest"


# =====================================
# UpscalerESRGAN
# =====================================
import sys

import numpy as np
import torch
from PIL import Image


def mod2normal(state_dict):
    # this code is copied from https://github.com/victorca25/iNNfer
    if "conv_first.weight" in state_dict:
        crt_net = {}
        items = list(state_dict)

        crt_net["model.0.weight"] = state_dict["conv_first.weight"]
        crt_net["model.0.bias"] = state_dict["conv_first.bias"]

        for k in items.copy():
            if "RDB" in k:
                ori_k = k.replace("RRDB_trunk.", "model.1.sub.")
                if ".weight" in k:
                    ori_k = ori_k.replace(".weight", ".0.weight")
                elif ".bias" in k:
                    ori_k = ori_k.replace(".bias", ".0.bias")
                crt_net[ori_k] = state_dict[k]
                items.remove(k)

        crt_net["model.1.sub.23.weight"] = state_dict["trunk_conv.weight"]
        crt_net["model.1.sub.23.bias"] = state_dict["trunk_conv.bias"]
        crt_net["model.3.weight"] = state_dict["upconv1.weight"]
        crt_net["model.3.bias"] = state_dict["upconv1.bias"]
        crt_net["model.6.weight"] = state_dict["upconv2.weight"]
        crt_net["model.6.bias"] = state_dict["upconv2.bias"]
        crt_net["model.8.weight"] = state_dict["HRconv.weight"]
        crt_net["model.8.bias"] = state_dict["HRconv.bias"]
        crt_net["model.10.weight"] = state_dict["conv_last.weight"]
        crt_net["model.10.bias"] = state_dict["conv_last.bias"]
        state_dict = crt_net
    return state_dict


def resrgan2normal(state_dict, nb=23):
    # this code is copied from https://github.com/victorca25/iNNfer
    if "conv_first.weight" in state_dict and "body.0.rdb1.conv1.weight" in state_dict:
        re8x = 0
        crt_net = {}
        items = list(state_dict)

        crt_net["model.0.weight"] = state_dict["conv_first.weight"]
        crt_net["model.0.bias"] = state_dict["conv_first.bias"]

        for k in items.copy():
            if "rdb" in k:
                ori_k = k.replace("body.", "model.1.sub.")
                ori_k = ori_k.replace(".rdb", ".RDB")
                if ".weight" in k:
                    ori_k = ori_k.replace(".weight", ".0.weight")
                elif ".bias" in k:
                    ori_k = ori_k.replace(".bias", ".0.bias")
                crt_net[ori_k] = state_dict[k]
                items.remove(k)

        crt_net[f"model.1.sub.{nb}.weight"] = state_dict["conv_body.weight"]
        crt_net[f"model.1.sub.{nb}.bias"] = state_dict["conv_body.bias"]
        crt_net["model.3.weight"] = state_dict["conv_up1.weight"]
        crt_net["model.3.bias"] = state_dict["conv_up1.bias"]
        crt_net["model.6.weight"] = state_dict["conv_up2.weight"]
        crt_net["model.6.bias"] = state_dict["conv_up2.bias"]

        if "conv_up3.weight" in state_dict:
            # modification supporting: https://github.com/ai-forever/Real-ESRGAN/blob/main/RealESRGAN/rrdbnet_arch.py
            re8x = 3
            crt_net["model.9.weight"] = state_dict["conv_up3.weight"]
            crt_net["model.9.bias"] = state_dict["conv_up3.bias"]

        crt_net[f"model.{8+re8x}.weight"] = state_dict["conv_hr.weight"]
        crt_net[f"model.{8+re8x}.bias"] = state_dict["conv_hr.bias"]
        crt_net[f"model.{10+re8x}.weight"] = state_dict["conv_last.weight"]
        crt_net[f"model.{10+re8x}.bias"] = state_dict["conv_last.bias"]

        state_dict = crt_net
    return state_dict


def infer_params(state_dict):
    # this code is copied from https://github.com/victorca25/iNNfer
    scale2x = 0
    scalemin = 6
    n_uplayer = 0
    plus = False

    for block in list(state_dict):
        parts = block.split(".")
        n_parts = len(parts)
        if n_parts == 5 and parts[2] == "sub":
            nb = int(parts[3])
        elif n_parts == 3:
            part_num = int(parts[1])
            if part_num > scalemin and parts[0] == "model" and parts[2] == "weight":
                scale2x += 1
            if part_num > n_uplayer:
                n_uplayer = part_num
                out_nc = state_dict[block].shape[0]
        if not plus and "conv1x1" in block:
            plus = True

    nf = state_dict["model.0.weight"].shape[0]
    in_nc = state_dict["model.0.weight"].shape[1]
    out_nc = out_nc
    scale = 2**scale2x

    return in_nc, out_nc, nf, nb, plus, scale


class UpscalerESRGAN(Upscaler):
    def __init__(self, tile=100, tile_overlap=10,  dirname=""):
        self.name = "ESRGAN"
        self.model_url = (
            "https://github.com/cszn/KAIR/releases/download/v1.0/ESRGAN.pth"
        )
        self.model_name = "ESRGAN_4x"
        self.user_path = dirname

        super().__init__()

        self.tile = tile
        self.tile_overlap = tile_overlap

    def do_upscale(self, img, selected_model):
        try:
            model = self.load_model(selected_model)
        except Exception as e:
            print(f"Unable to load ESRGAN model {selected_model}: {e}", file=sys.stderr)
            return img
        model.to(device_upscaler)
        img = esrgan_upscale(model, img, self.tile, self.tile_overlap)
        return img

    def load_model(self, path: str):
        if path.startswith("http"):
            filename = load_file_from_url(
                url=path,
                model_dir=self.model_download_path,
                file_name=f"{friendly_name(path)}.pth",
            )
        else:
            filename = path

        state_dict = torch.load(
            filename, map_location="cpu"
        )

        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]
            num_conv = 16 if "realesr-animevideov3" in filename else 32
            model = SRVGGNetCompact(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_conv=num_conv,
                upscale=4,
                act_type="prelu",
            )
            model.load_state_dict(state_dict)
            model.eval()
            return model

        if (
            "body.0.rdb1.conv1.weight" in state_dict
            and "conv_first.weight" in state_dict
        ):
            nb = 6 if "RealESRGAN_x4plus_anime_6B" in filename else 23
            state_dict = resrgan2normal(state_dict, nb)
        elif "conv_first.weight" in state_dict:
            state_dict = mod2normal(state_dict)
        elif "model.0.weight" not in state_dict:
            raise Exception("The file is not a recognized ESRGAN model.")

        in_nc, out_nc, nf, nb, plus, mscale = infer_params(state_dict)

        model = RRDBNet(
            in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb, upscale=mscale, plus=plus
        )
        model.load_state_dict(state_dict)
        model.eval()

        return model


def upscale_without_tiling(model, img):
    img = np.array(img)
    img = img[:, :, ::-1]
    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1))) / 255
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0).to(device_upscaler)
    with torch.no_grad():
        output = model(img)
    output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = 255.0 * np.moveaxis(output, 0, 2)
    output = output.astype(np.uint8)
    output = output[:, :, ::-1]
    return Image.fromarray(output, "RGB")


def esrgan_upscale(model, img, ESRGAN_tile, ESRGAN_tile_overlap):
    if ESRGAN_tile == 0:  # no tiling
        return upscale_without_tiling(model, img)
    # ESRGAN_tile = 100
    # ESRGAN_tile_overlap = 10

    grid = split_grid(img, ESRGAN_tile, ESRGAN_tile, ESRGAN_tile_overlap)
    newtiles = []
    scale_factor = 1 # 4

    for y, h, row in grid.tiles:
        newrow = []
        for tiledata in row:
            x, w, tile = tiledata

            output = upscale_without_tiling(model, tile)
            scale_factor = output.width // tile.width

            newrow.append([x * scale_factor, w * scale_factor, output])
        newtiles.append([y * scale_factor, h * scale_factor, newrow])

    newgrid = Grid(
        newtiles,
        grid.tile_w * scale_factor,
        grid.tile_h * scale_factor,
        grid.image_w * scale_factor,
        grid.image_h * scale_factor,
        grid.overlap * scale_factor,
    )
    output = combine_grid(newgrid)
    return output


# =====================================
# Upscaler Utils
# =====================================

from collections import namedtuple

Grid = namedtuple(
    "Grid", ["tiles", "tile_w", "tile_h", "image_w", "image_h", "overlap"]
)


def split_grid(image, tile_w=512, tile_h=512, overlap=64):
    w = image.width
    h = image.height

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
        return Image.fromarray(r, "L")

    mask_w = make_mask_image(
        np.arange(grid.overlap, dtype=np.float32)
        .reshape((1, grid.overlap))
        .repeat(grid.tile_h, axis=0)
    )
    mask_h = make_mask_image(
        np.arange(grid.overlap, dtype=np.float32)
        .reshape((grid.overlap, 1))
        .repeat(grid.image_w, axis=1)
    )

    combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
    for y, h, row in grid.tiles:
        combined_row = Image.new("RGB", (grid.image_w, h))
        for x, w, tile in row:
            if x == 0:
                combined_row.paste(tile, (0, 0))
                continue

            combined_row.paste(tile.crop((0, 0, grid.overlap, h)), (x, 0), mask=mask_w)
            combined_row.paste(
                tile.crop((grid.overlap, 0, w, h)), (x + grid.overlap, 0)
            )

        if y == 0:
            combined_image.paste(combined_row, (0, 0))
            continue

        combined_image.paste(
            combined_row.crop((0, 0, combined_row.width, grid.overlap)),
            (0, y),
            mask=mask_h,
        )
        combined_image.paste(
            combined_row.crop((0, grid.overlap, combined_row.width, h)),
            (0, y + grid.overlap),
        )

    return combined_image


# =====================================
# Upscaler loaders
# =====================================
import os
from urllib.parse import urlparse


def load_file_from_url(
    url: str,
    *,
    model_dir: str,
    progress: bool = True,
    file_name: str | None = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file

        download_url_to_file(url, cached_file, progress=progress)
    return cached_file

def friendly_name(file: str):
    if file.startswith("http"):
        file = urlparse(file).path

    file = os.path.basename(file)
    model_name, extension = os.path.splitext(file)
    return model_name


# =====================================
# UpscalerESRGAN ARCH
# =====================================

# this file is adapted from https://github.com/victorca25/iNNfer

from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


####################
# RRDBNet Generator
####################


class RRDBNet(nn.Module):
    def __init__(
        self,
        in_nc,
        out_nc,
        nf,
        nb,
        nr=3,
        gc=32,
        upscale=4,
        norm_type=None,
        act_type="leakyrelu",
        mode="CNA",
        upsample_mode="upconv",
        convtype="Conv2D",
        finalact=None,
        gaussian_noise=False,
        plus=False,
    ):
        super(RRDBNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        self.resrgan_scale = 0
        if in_nc % 16 == 0:
            self.resrgan_scale = 1
        elif in_nc != 4 and in_nc % 4 == 0:
            self.resrgan_scale = 2

        fea_conv = conv_block(
            in_nc, nf, kernel_size=3, norm_type=None, act_type=None, convtype=convtype
        )
        rb_blocks = [
            RRDB(
                nf,
                nr,
                kernel_size=3,
                gc=32,
                stride=1,
                bias=1,
                pad_type="zero",
                norm_type=norm_type,
                act_type=act_type,
                mode="CNA",
                convtype=convtype,
                gaussian_noise=gaussian_noise,
                plus=plus,
            )
            for _ in range(nb)
        ]
        LR_conv = conv_block(
            nf,
            nf,
            kernel_size=3,
            norm_type=norm_type,
            act_type=None,
            mode=mode,
            convtype=convtype,
        )

        if upsample_mode == "upconv":
            upsample_block = upconv_block
        elif upsample_mode == "pixelshuffle":
            upsample_block = pixelshuffle_block
        else:
            raise NotImplementedError(f"upsample mode [{upsample_mode}] is not found")
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type, convtype=convtype)
        else:
            upsampler = [
                upsample_block(nf, nf, act_type=act_type, convtype=convtype)
                for _ in range(n_upscale)
            ]
        HR_conv0 = conv_block(
            nf, nf, kernel_size=3, norm_type=None, act_type=act_type, convtype=convtype
        )
        HR_conv1 = conv_block(
            nf, out_nc, kernel_size=3, norm_type=None, act_type=None, convtype=convtype
        )

        outact = act(finalact) if finalact else None

        self.model = sequential(
            fea_conv,
            ShortcutBlock(sequential(*rb_blocks, LR_conv)),
            *upsampler,
            HR_conv0,
            HR_conv1,
            outact,
        )

    def forward(self, x, outm=None):
        if self.resrgan_scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        elif self.resrgan_scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        else:
            feat = x

        return self.model(feat)


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    """

    def __init__(
        self,
        nf,
        nr=3,
        kernel_size=3,
        gc=32,
        stride=1,
        bias=1,
        pad_type="zero",
        norm_type=None,
        act_type="leakyrelu",
        mode="CNA",
        convtype="Conv2D",
        spectral_norm=False,
        gaussian_noise=False,
        plus=False,
    ):
        super(RRDB, self).__init__()
        # This is for backwards compatibility with existing models
        if nr == 3:
            self.RDB1 = ResidualDenseBlock_5C(
                nf,
                kernel_size,
                gc,
                stride,
                bias,
                pad_type,
                norm_type,
                act_type,
                mode,
                convtype,
                spectral_norm=spectral_norm,
                gaussian_noise=gaussian_noise,
                plus=plus,
            )
            self.RDB2 = ResidualDenseBlock_5C(
                nf,
                kernel_size,
                gc,
                stride,
                bias,
                pad_type,
                norm_type,
                act_type,
                mode,
                convtype,
                spectral_norm=spectral_norm,
                gaussian_noise=gaussian_noise,
                plus=plus,
            )
            self.RDB3 = ResidualDenseBlock_5C(
                nf,
                kernel_size,
                gc,
                stride,
                bias,
                pad_type,
                norm_type,
                act_type,
                mode,
                convtype,
                spectral_norm=spectral_norm,
                gaussian_noise=gaussian_noise,
                plus=plus,
            )
        else:
            RDB_list = [
                ResidualDenseBlock_5C(
                    nf,
                    kernel_size,
                    gc,
                    stride,
                    bias,
                    pad_type,
                    norm_type,
                    act_type,
                    mode,
                    convtype,
                    spectral_norm=spectral_norm,
                    gaussian_noise=gaussian_noise,
                    plus=plus,
                )
                for _ in range(nr)
            ]
            self.RDBs = nn.Sequential(*RDB_list)

    def forward(self, x):
        if hasattr(self, "RDB1"):
            out = self.RDB1(x)
            out = self.RDB2(out)
            out = self.RDB3(out)
        else:
            out = self.RDBs(x)
        return out * 0.2 + x


class ResidualDenseBlock_5C(nn.Module):
    """
    Residual Dense Block
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    Modified options that can be used:
        - "Partial Convolution based Padding" arXiv:1811.11718
        - "Spectral normalization" arXiv:1802.05957
        - "ICASSP 2020 - ESRGAN+ : Further Improving ESRGAN" N. C.
            {Rakotonirina} and A. {Rasoanaivo}
    """

    def __init__(
        self,
        nf=64,
        kernel_size=3,
        gc=32,
        stride=1,
        bias=1,
        pad_type="zero",
        norm_type=None,
        act_type="leakyrelu",
        mode="CNA",
        convtype="Conv2D",
        spectral_norm=False,
        gaussian_noise=False,
        plus=False,
    ):
        super(ResidualDenseBlock_5C, self).__init__()

        self.noise = GaussianNoise() if gaussian_noise else None
        self.conv1x1 = conv1x1(nf, gc) if plus else None

        self.conv1 = conv_block(
            nf,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            convtype=convtype,
            spectral_norm=spectral_norm,
        )
        self.conv2 = conv_block(
            nf + gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            convtype=convtype,
            spectral_norm=spectral_norm,
        )
        self.conv3 = conv_block(
            nf + 2 * gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            convtype=convtype,
            spectral_norm=spectral_norm,
        )
        self.conv4 = conv_block(
            nf + 3 * gc,
            gc,
            kernel_size,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=act_type,
            mode=mode,
            convtype=convtype,
            spectral_norm=spectral_norm,
        )
        if mode == "CNA":
            last_act = None
        else:
            last_act = act_type
        self.conv5 = conv_block(
            nf + 4 * gc,
            nf,
            3,
            stride,
            bias=bias,
            pad_type=pad_type,
            norm_type=norm_type,
            act_type=last_act,
            mode=mode,
            convtype=convtype,
            spectral_norm=spectral_norm,
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        if self.conv1x1:
            x2 = x2 + self.conv1x1(x)
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        if self.conv1x1:
            x4 = x4 + x2
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        if self.noise:
            return self.noise(x5.mul(0.2) + x)
        else:
            return x5 * 0.2 + x


####################
# ESRGANplus
####################


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1, is_relative_detach=False):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0, dtype=torch.float)

    def forward(self, x):
        if self.training and self.sigma != 0:
            self.noise = self.noise.to(x.device)
            scale = (
                self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            )
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


####################
# SRVGGNetCompact
####################


class SRVGGNetCompact(nn.Module):
    """A compact VGG-style network structure for super-resolution.
    This class is copied from https://github.com/xinntao/Real-ESRGAN
    """

    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=16,
        upscale=4,
        act_type="prelu",
    ):
        super(SRVGGNetCompact, self).__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # the first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # the first activation
        if act_type == "relu":
            activation = nn.ReLU(inplace=True)
        elif act_type == "prelu":
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == "leakyrelu":
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        # the body structure
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            # activation
            if act_type == "relu":
                activation = nn.ReLU(inplace=True)
            elif act_type == "prelu":
                activation = nn.PReLU(num_parameters=num_feat)
            elif act_type == "leakyrelu":
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.body.append(activation)

        # the last conv
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        # upsample
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for i in range(0, len(self.body)):
            out = self.body[i](out)

        out = self.upsampler(out)
        # add the nearest upsampled image, so that the network learns the residual
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        out += base
        return out


####################
# Upsampler
####################


class Upsample(nn.Module):
    r"""Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.
    The input data is assumed to be of the form
    `minibatch x channels x [optional depth] x [optional height] x width`.
    """

    def __init__(
        self, size=None, scale_factor=None, mode="nearest", align_corners=None
    ):
        super(Upsample, self).__init__()
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.size = size
        self.align_corners = align_corners

    def forward(self, x):
        return nn.functional.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

    def extra_repr(self):
        if self.scale_factor is not None:
            info = f"scale_factor={self.scale_factor}"
        else:
            info = f"size={self.size}"
        info += f", mode={self.mode}"
        return info


def pixel_unshuffle(x, scale):
    """Pixel unshuffle.
    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.
    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


def pixelshuffle_block(
    in_nc,
    out_nc,
    upscale_factor=2,
    kernel_size=3,
    stride=1,
    bias=True,
    pad_type="zero",
    norm_type=None,
    act_type="relu",
    convtype="Conv2D",
):
    """
    Pixel shuffle layer
    (Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional
    Neural Network, CVPR17)
    """
    conv = conv_block(
        in_nc,
        out_nc * (upscale_factor**2),
        kernel_size,
        stride,
        bias=bias,
        pad_type=pad_type,
        norm_type=None,
        act_type=None,
        convtype=convtype,
    )
    pixel_shuffle = nn.PixelShuffle(upscale_factor)

    n = norm(norm_type, out_nc) if norm_type else None
    a = act(act_type) if act_type else None
    return sequential(conv, pixel_shuffle, n, a)


def upconv_block(
    in_nc,
    out_nc,
    upscale_factor=2,
    kernel_size=3,
    stride=1,
    bias=True,
    pad_type="zero",
    norm_type=None,
    act_type="relu",
    mode="nearest",
    convtype="Conv2D",
):
    """Upconv layer"""
    upscale_factor = (
        (1, upscale_factor, upscale_factor) if convtype == "Conv3D" else upscale_factor
    )
    upsample = Upsample(scale_factor=upscale_factor, mode=mode)
    conv = conv_block(
        in_nc,
        out_nc,
        kernel_size,
        stride,
        bias=bias,
        pad_type=pad_type,
        norm_type=norm_type,
        act_type=act_type,
        convtype=convtype,
    )
    return sequential(upsample, conv)


####################
# Basic blocks
####################


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        basic_block (nn.module): nn.module class for basic block. (block)
        num_basic_block (int): number of blocks. (n_layers)
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


def act(act_type, inplace=True, neg_slope=0.2, n_prelu=1, beta=1.0):
    """activation helper"""
    act_type = act_type.lower()
    if act_type == "relu":
        layer = nn.ReLU(inplace)
    elif act_type in ("leakyrelu", "lrelu"):
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act_type == "tanh":  # [-1, 1] range output
        layer = nn.Tanh()
    elif act_type == "sigmoid":  # [0, 1] range output
        layer = nn.Sigmoid()
    else:
        raise NotImplementedError(f"activation layer [{act_type}] is not found")
    return layer


class Identity(nn.Module):
    def __init__(self, *kwargs):
        super(Identity, self).__init__()

    def forward(self, x, *kwargs):
        return x


def norm(norm_type, nc):
    """Return a normalization layer"""
    norm_type = norm_type.lower()
    if norm_type == "batch":
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == "instance":
        layer = nn.InstanceNorm2d(nc, affine=False)
    elif norm_type == "none":

        def norm_layer(x):
            return Identity()

    else:
        raise NotImplementedError(f"normalization layer [{norm_type}] is not found")
    return layer


def pad(pad_type, padding):
    """padding layer helper"""
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == "reflect":
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == "replicate":
        layer = nn.ReplicationPad2d(padding)
    elif pad_type == "zero":
        layer = nn.ZeroPad2d(padding)
    else:
        raise NotImplementedError(f"padding layer [{pad_type}] is not implemented")
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


class ShortcutBlock(nn.Module):
    """Elementwise sum the output of a submodule to its input"""

    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

    def __repr__(self):
        return "Identity + \n|" + self.sub.__repr__().replace("\n", "\n|")


def sequential(*args):
    """Flatten Sequential. It unwraps nn.Sequential."""
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def conv_block(
    in_nc,
    out_nc,
    kernel_size,
    stride=1,
    dilation=1,
    groups=1,
    bias=True,
    pad_type="zero",
    norm_type=None,
    act_type="relu",
    mode="CNA",
    convtype="Conv2D",
    spectral_norm=False,
):
    """Conv layer with padding, normalization, activation"""
    assert mode in ["CNA", "NAC", "CNAC"], f"Wrong conv mode [{mode}]"
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != "zero" else None
    padding = padding if pad_type == "zero" else 0

    if convtype == "PartialConv2D":
        from torchvision.ops import (
            PartialConv2d,
        )  # this is definitely not going to work, but PartialConv2d doesn't work anyway and this shuts up static analyzer

        c = PartialConv2d(
            in_nc,
            out_nc,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=groups,
        )
    elif convtype == "DeformConv2D":
        from torchvision.ops import DeformConv2d  # not tested

        c = DeformConv2d(
            in_nc,
            out_nc,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=groups,
        )
    elif convtype == "Conv3D":
        c = nn.Conv3d(
            in_nc,
            out_nc,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=groups,
        )
    else:
        c = nn.Conv2d(
            in_nc,
            out_nc,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=groups,
        )

    if spectral_norm:
        c = nn.utils.spectral_norm(c)

    a = act(act_type) if act_type else None
    if "CNA" in mode:
        n = norm(norm_type, out_nc) if norm_type else None
        return sequential(p, c, n, a)
    elif mode == "NAC":
        if norm_type is None and act_type is not None:
            a = act(act_type, inplace=False)
        n = norm(norm_type, in_nc) if norm_type else None
        return sequential(n, a, p, c)
