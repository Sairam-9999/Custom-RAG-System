"""Download and load OpenAI GPT-2 checkpoints for the local PyTorch GPT model."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import requests
import tensorflow as tf
from tqdm import tqdm


def download_and_load_gpt2(model_size: str = "124M", models_dir: str = "gpt2"):
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"model_size must be one of {allowed_sizes}")

    model_dir = Path(models_dir) / model_size
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]

    model_dir.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        url = f"{base_url}/{model_size}/{filename}"
        destination = model_dir / filename
        download_file(url, destination)

    tf_ckpt_path = tf.train.latest_checkpoint(str(model_dir))
    if tf_ckpt_path is None:
        raise FileNotFoundError(f"No TensorFlow checkpoint found in {model_dir}")

    with open(model_dir / "hparams.json", "r", encoding="utf-8") as f:
        settings = json.load(f)

    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)
    return settings, params


def download_file(url: str, destination: str | Path, chunk_size: int = 1024) -> None:
    destination = Path(destination)

    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()

    file_size = int(response.headers.get("content-length", 0))
    if destination.exists() and file_size and destination.stat().st_size == file_size:
        print(f"File already exists and is up-to-date: {destination}")
        return

    tmp_destination = destination.with_suffix(destination.suffix + ".tmp")
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=destination.name) as progress_bar:
        with open(tmp_destination, "wb") as f:
            for chunk in response.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))

    os.replace(tmp_destination, destination)


def load_gpt2_params_from_tf_ckpt(ckpt_path: str, settings: dict) -> dict:
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    for name, _ in tf.train.list_variables(ckpt_path):
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        variable_name_parts = name.split("/")[1:]  # remove "model" prefix

        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        target_dict[variable_name_parts[-1]] = variable_array

    return params


if __name__ == "__main__":
    settings, params = download_and_load_gpt2("124M", "gpt2")
    print("Downloaded and loaded GPT-2 settings:")
    print(settings)
    print("Top-level params:", params.keys())
