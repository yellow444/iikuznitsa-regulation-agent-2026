$ErrorActionPreference = 'Stop'
$source = 'https://huggingface.co/Xenova/multilingual-e5-small/resolve/761b726'
$modelFolder = New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot 'runtime/e5')
$files = [ordered]@{'tokenizer.json' = 'tokenizer.json'; 'model.onnx' = 'onnx/model_quantized.onnx'}

foreach ($name in $files.Keys) {
    $target = Join-Path $modelFolder.FullName $name
    if (Test-Path -LiteralPath $target -PathType Leaf) { continue }
    $staging = "$target.partial"
    curl.exe -fLsS --retry 1 --max-time 120 -o $staging "$source/$($files[$name])"
    if ($LASTEXITCODE -ne 0) { throw "Не удалось скачать $name" }
    Move-Item -LiteralPath $staging -Destination $target
}
