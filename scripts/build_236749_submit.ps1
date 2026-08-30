param(
    [string]$OutputName = 'submit.zip'
)

$ErrorActionPreference = 'Stop'

$workspace = Split-Path -Parent $PSScriptRoot
$runtime = Join-Path $workspace 'model\runtime'
$runtimeSource = Join-Path $runtime 'src'
$source = Join-Path $workspace 'src'

if ([IO.Path]::GetExtension($OutputName) -ne '.zip') {
    throw "OutputName must end in .zip"
}
$archive = Join-Path $workspace $OutputName

# The contest rejects an extra top-level source directory. Bundle importable
# source under the permitted model/ directory instead.
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
New-Item -ItemType Directory -Force -Path $runtimeSource | Out-Null
Copy-Item -Path (Join-Path $source '*') -Destination $runtimeSource -Recurse -Force -Exclude '__pycache__', '*.pyc'

Push-Location $workspace
try {
    # Windows Compress-Archive can fail on the 1.37 GB ONNX file when it is
    # memory-mapped. libarchive's tar.exe writes the same ZIP layout reliably.
    tar.exe --exclude='model/runtime/src/src' --exclude='model/df_arena/*.ort' --exclude='model/df_arena/.cache' --exclude='model/aasist_64.pt' --exclude='model/aasist_best.pt' --exclude='model/heuristic.pt' --exclude='model/music_spec_cnn.pt' --exclude='model/stageA_aasist.pt' --exclude='model/voice_aasist.pt' --exclude='model/voice_spec_cnn.pt' --exclude='*__pycache__*' --exclude='*.pyc' -a -c -f $archive model script.py requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "tar.exe failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}

Write-Output "Created $archive"
