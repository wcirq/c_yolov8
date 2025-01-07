import struct
import argparse
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Conv


# -----------------------------------------------------------------------------
# common utilities

def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    size = tensor.numel()
    file.write(struct.pack(f'i', size))

    shape = tensor.size()
    if len(shape) == 1: shape = [shape[0], 0, 0, 0]
    elif len(shape) == 2: shape = [shape[0], shape[1], 0, 0]
    elif len(shape) == 3: shape = [shape[0], shape[1], shape[2], 0]
    elif len(shape) == 4: shape = [shape[0], shape[1], shape[2], shape[3]]
    else: raise Exception("不支持大于4维的tensor")
    file.write(struct.pack(f'iiii', *shape))

    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)


def serialize_int8(file, tensor):
    """ writes one int8 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).numpy().astype(np.int8)
    b = struct.pack(f'{len(d)}b', *d)
    file.write(b)


def quantize_q80(w, group_size):
    """
    takes a tensor and returns the Q8_0 quantized version
    i.e. symmetric quantization into int8, range [-127,127]
    """
    assert w.numel() % group_size == 0
    ori_shape = w.shape
    w = w.float()  # convert to float32
    w = w.reshape(-1, group_size)
    # find the max in each group
    wmax = torch.abs(w).max(dim=1).values
    # calculate the scaling factor such that float = quant * scale
    scale = wmax / 127.0
    # scale into range [-127, 127]
    quant = w / scale[:, None]
    # round to nearest integer
    int8val = torch.round(quant).to(torch.int8)
    # dequantize by rescaling
    fp32val = (int8val.float() * scale[:, None]).view(-1)
    fp32valr = fp32val.reshape(-1, group_size)
    # calculate the max error in each group
    err = torch.abs(fp32valr - w).max(dim=1).values
    # find the max error across all groups
    maxerr = err.max().item()
    return int8val, scale, maxerr


# -----------------------------------------------------------------------------
# legacy

def legacy_export(model, filepath):
    """ Original export of llama2.c bin files, i.e. version v0 """
    out_file = open(filepath, 'wb')

    layers = list(model.model.model)

    header = struct.pack('i', len(layers))
    out_file.write(header)

    # now all the layers
    for layer in layers[:1]:
        if isinstance(layer, Conv):
            conv_weight = layer.conv.weight          # ω
            conv_bias = layer.conv.bias              # b
            bn_weight = layer.bn.weight              # γ
            bn_bias = layer.bn.bias                  # β
            bn_running_var = layer.bn.running_var    # σ^2
            bn_eps = layer.bn.eps                    # ε
            bn_running_mean = layer.bn.running_mean  # μ
            # 处理 bias
            conv_bias = torch.zeros(conv_weight.shape[0], device=conv_weight.device) if conv_bias is None else conv_bias

            serialize_fp32(out_file, conv_weight)  # shape 48 3 3 3
            serialize_fp32(out_file, conv_bias)  # shape 48
            serialize_fp32(out_file, bn_weight)  # shape 48
            serialize_fp32(out_file, bn_bias)  # shape 48
            serialize_fp32(out_file, bn_running_var)  # shape 48
            serialize_fp32(out_file, bn_running_mean)  # shape 48

            bn_eps_bytes = struct.pack('f', bn_eps)  # value 0.0001
            out_file.write(bn_eps_bytes)

    # write to binary file
    out_file.close()
    print(f"wrote {filepath}")


def load_checkpoint(checkpoint):
    # load the provided model checkpoint
    # checkpoint_dict = torch.load(checkpoint, map_location='cpu')
    model = YOLO(checkpoint)  # build from YAML and transfer weights
    return model


def model_export(model, filepath, version, dtype=torch.float32):
    """
    Versions docs:
    v-1:huggingface export, i.e. intended for use outside of this repo, in HF
    v0: legacy llama2.c float format, DEPRECATED
    v1: float32 export
    v2: int8 quantized Q8_0 export, similar to llama.cpp, in groups
    # TODO: add dtype export support for other versions (?)
    """
    if version == 0:
        legacy_export(model, filepath)
    else:
        raise ValueError(f"unknown version {version}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="the output filepath")
    parser.add_argument("--version", default=0, type=int, help="the version to export with")
    parser.add_argument("--dtype", type=str, help="dtype of the model (fp16, fp32)", default="fp32")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="model checkpoint, .pt file")
    args = parser.parse_args()
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    model = None
    if args.checkpoint:
        model = load_checkpoint(args.checkpoint)

    if model is None:
        parser.error("Can't load input model!")

    # export
    model_export(model, args.filepath, args.version, args.dtype)
