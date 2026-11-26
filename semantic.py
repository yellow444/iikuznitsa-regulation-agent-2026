from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


class Encoder:
    def __init__(self, folder: Path):
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(folder / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.inputs = {value.name for value in self.session.get_inputs()}

    def encode(self, values: list[str], prefix: str) -> np.ndarray:
        chunks = []
        for start in range(0, len(values), 16):
            batch = self.tokenizer.encode_batch(
                [f"{prefix}: {value}" for value in values[start:start + 16]]
            )
            arrays = {
                "input_ids": np.array([item.ids for item in batch], dtype=np.int64),
                "attention_mask": np.array([item.attention_mask for item in batch], dtype=np.int64),
                "token_type_ids": np.array([item.type_ids for item in batch], dtype=np.int64),
            }
            hidden = self.session.run(None, {name: arrays[name] for name in self.inputs})[0]
            mask = arrays["attention_mask"][..., None].astype(np.float32)
            pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1)
            chunks.append(pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12))
        return np.concatenate(chunks, axis=0)
