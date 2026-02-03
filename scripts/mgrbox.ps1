Param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

$ImageName = $env:IMAGE_NAME
if (-not $ImageName) { $ImageName = "mgrcli:local" }

$DefaultOutput = Join-Path $HOME "MGR_Results"
$OutputDir = $env:MGR_RESULTS_DIR
if (-not $OutputDir) { $OutputDir = $DefaultOutput }

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$RayonThreads = $env:RAYON_NUM_THREADS
if (-not $RayonThreads) {
  $RayonThreads = $env:MGR_RAYON_THREADS
}
if (-not $RayonThreads) {
  $memTotal = (docker info --format "{{.MemTotal}}" 2>$null)
  if ($memTotal) {
    $gib = 1024 * 1024 * 1024
    if ($memTotal -le 6 * $gib) { $RayonThreads = 2 }
    elseif ($memTotal -le 10 * $gib) { $RayonThreads = 4 }
    elseif ($memTotal -le 14 * $gib) { $RayonThreads = 6 }
    elseif ($memTotal -le 20 * $gib) { $RayonThreads = 8 }
    else { $RayonThreads = 10 }
  } else {
    $RayonThreads = 4
  }
}

docker run --rm `
  -v "${PWD}:/work" `
  -v "${OutputDir}:/results" `
  -w /work `
  -e MGR_RESULTS_DIR=/results `
  -e RAYON_NUM_THREADS=$RayonThreads `
  $ImageName `
  python -m mgrcli.cli @Args
