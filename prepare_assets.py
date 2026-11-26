"""Fetch the organizer's public sample files and the pinned local encoder."""

from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from shutil import copyfileobj
from urllib.request import Request, urlopen

from retrieval import SOURCES


DOCUMENT_HASHES = {
    "dbrct.docx": "be8a86386276768352e67cfeaef807f98b79f8532bef7c649981326184d4acbd",
    "ib-presales.docx": "e39235e9a4257c0cb55ccb557198cb9e0f4313d1c4e4e797cdefacc5e279b7f3",
    "ib-project.docx": "35c9d64940273d19fc27f79e6436056b5944ea975533c53579222c049056b9b4",
    "tech.docx": "d0a84128c45ba50eccd6715ae534bba0642f1a1b5af1468269d5ad8e079b5dbf",
}
MODEL_ORIGIN = "https://huggingface.co/Xenova/multilingual-e5-small/resolve/761b726"
MODEL_FILES = {
    "tokenizer.json": ("tokenizer.json", 17_082_730),
    "model.onnx": ("onnx/model_quantized.onnx", 118_308_185),
}


def fetch(url, target, valid):
    if target.is_file() and valid(target):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_suffix(target.suffix + ".part")
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 IikuznitsaDemo/1.0"})
    with urlopen(request, timeout=120) as response, staged.open("wb") as output:
        copyfileobj(response, output)
    if not valid(staged):
        raise ValueError(f"Unexpected content for {target.name}")
    staged.replace(target)


def prepare_corpus(corpus):
    for filename, (_, file_id) in SOURCES.items():
        fetch(
            f"https://drive.google.com/uc?export=download&id={file_id}",
            corpus / filename,
            lambda path: sha256(path.read_bytes()).hexdigest() == DOCUMENT_HASHES[filename],
        )


def prepare_model(model):
    for filename, (remote, size) in MODEL_FILES.items():
        fetch(f"{MODEL_ORIGIN}/{remote}", model / filename, lambda path: path.stat().st_size == size)


def split_embeddings(model):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    # Move the quantized lookup table to a file that the runtime can memory-map.
    model_path = model / "model.onnx"
    encoder = onnx.load(model_path)
    graph = encoder.graph
    tensor_name = "embeddings.word_embeddings.weight_quantized"
    weights = next(tensor for tensor in graph.initializer if tensor.name == tensor_name)
    gather, = (node for node in graph.node if tensor_name in node.input)
    lookup_input = "/embeddings/word_embeddings/Gather_output_0_quantized"
    if (
        gather.op_type != "Gather"
        or list(gather.input) != [tensor_name, "input_ids"]
        or list(gather.output) != [lookup_input]
        or any((attr.name, attr.i) != ("axis", 0) for attr in gather.attribute)
    ):
        raise ValueError("Unexpected graph layout for the word embedding lookup.")
    table = numpy_helper.to_array(weights)
    if table.dtype != np.uint8 or table.ndim != 2:
        raise ValueError("Unexpected word embedding table type or shape.")
    graph.input.append(helper.make_tensor_value_info(
        lookup_input, TensorProto.UINT8, ["batch_size", "sequence_length", table.shape[1]]
    ))
    graph.node.remove(gather)
    graph.initializer.remove(weights)
    onnx.checker.check_model(encoder)
    staged_model = model_path.with_suffix(".part")
    np.save(model / "word_embeddings.npy", table, allow_pickle=False)
    onnx.save(encoder, staged_model)
    staged_model.replace(model_path)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--split-embeddings", action="store_true")
    args = parser.parse_args()
    if not (args.corpus or args.model):
        parser.error("Specify --corpus, --model or both.")
    if args.split_embeddings and not args.model:
        parser.error("--split-embeddings requires --model.")
    if args.corpus:
        prepare_corpus(args.corpus)
    if args.model:
        prepare_model(args.model)
        if args.split_embeddings:
            split_embeddings(args.model)
