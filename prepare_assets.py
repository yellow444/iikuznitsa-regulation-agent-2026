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


def main(corpus, model):
    for filename, (_, file_id) in SOURCES.items():
        fetch(
            f"https://drive.google.com/uc?export=download&id={file_id}",
            corpus / filename,
            lambda path: sha256(path.read_bytes()).hexdigest() == DOCUMENT_HASHES[filename],
        )
    for filename, (remote, size) in MODEL_FILES.items():
        fetch(f"{MODEL_ORIGIN}/{remote}", model / filename, lambda path: path.stat().st_size == size)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()
    main(args.corpus, args.model)
