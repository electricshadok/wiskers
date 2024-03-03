import csv
import os
import time

import pandas as pd
import torch
import torch.nn as nn

from wiskers.common.modules.attentions_2d import SelfMultiheadAttention2D, SelfScaledDotProductAttention2D
from wiskers.common.modules.cbam import CBAM
from wiskers.common.modules.se_block_2d import SEBlock


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def calculate_model_size(model: nn.Module) -> float:
    """
    Calculates the size of a PyTorch model in bytes.
    """
    total_size_bytes = 0
    for param in model.parameters():
        total_size_bytes += param.nelement() * param.element_size()
    return total_size_bytes


def benchmark_self_attentions(input_shape=(1, 32, 128, 128), reduction_ratio=4):
    print("Start benchmark ...")
    device = get_device()
    in_channels = input_shape[1]
    cbam_att = CBAM(in_channels=in_channels, reduction_ratio=reduction_ratio)
    se_att = SEBlock(in_channels=in_channels, squeeze_channels=in_channels // reduction_ratio)
    multihead_att = SelfMultiheadAttention2D(channels=in_channels, num_heads=8)
    scaled_dot_att = SelfScaledDotProductAttention2D(channels=in_channels)

    dummy_input = torch.randn(input_shape, device=device)

    attentions = {
        "wiskers.common.modules.cbam": cbam_att,
        "wiskers.common.modules.squeeze-and-excitation": se_att,
        "wiskers.common.modules.multi_head(8)": multihead_att,
        "wiskers.common.modules.scaled_dot": scaled_dot_att,
    }

    # Perform benchmarks on the different modules
    cvs_headers = ["module_name", "inference_time", "bytes"]
    benchmark_data = []
    for name, att in attentions.items():
        att = att.to(device)
        with torch.no_grad():
            start_time = time.time()
            _ = att(dummy_input)
            elapsed_time = round(time.time() - start_time, 2)
            total_size_bytes = calculate_model_size(att)
            benchmark = {cvs_headers[0]: name, cvs_headers[1]: elapsed_time, cvs_headers[2]: total_size_bytes}
            benchmark_data.append(benchmark)

    # Export file
    csv_filepath = os.path.join(os.path.dirname(__file__), "benchmark_results.csv")
    with open(csv_filepath, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=cvs_headers)
        writer.writeheader()
        writer.writerows(benchmark_data)

    # Print CSV in from exported benchmark
    df = pd.read_csv(csv_filepath)
    print(df)

    print("End benchmark")


def main():
    input_shape = (1, 32, 128, 128)
    reduction_ratio = 4

    benchmark_self_attentions(input_shape, reduction_ratio)


if __name__ == "__main__":
    main()
