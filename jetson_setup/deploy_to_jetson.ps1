[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$JetsonIp,
    [string]$JetsonUser = "jetson",
    [string]$RemoteRoot = "~/robotics_exp1"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$remote = "$JetsonUser@$JetsonIp"

$files = @(
    @{ Local = Join-Path $projectRoot "weights\best.pt"; Remote = "$RemoteRoot/weights/best.pt" },
    @{ Local = Join-Path $projectRoot "realtime_detect.py"; Remote = "$RemoteRoot/code/realtime_detect.py" },
    @{ Local = Join-Path $projectRoot "ros2_detector_node.py"; Remote = "$RemoteRoot/code/ros2_detector_node.py" }
)

foreach ($file in $files) {
    if (-not (Test-Path -LiteralPath $file.Local -PathType Leaf)) {
        throw "Missing local file: $($file.Local)"
    }
}

Write-Host "Testing SSH connection to $remote ..."
& ssh $remote "echo SSH_OK; mkdir -p $RemoteRoot/weights $RemoteRoot/code $RemoteRoot/results"
if ($LASTEXITCODE -ne 0) { throw "SSH setup failed" }

foreach ($file in $files) {
    Write-Host "Copying $($file.Local)"
    & scp $file.Local "${remote}:$($file.Remote)"
    if ($LASTEXITCODE -ne 0) { throw "Copy failed: $($file.Local)" }
}

$localHash = (Get-FileHash (Join-Path $projectRoot "weights\best.pt") -Algorithm SHA256).Hash
$remoteHash = (& ssh $remote "sha256sum $RemoteRoot/weights/best.pt").Trim()
Write-Host "Local SHA256 : $localHash"
Write-Host "Remote SHA256: $remoteHash"
if ($remoteHash -notmatch $localHash) {
    throw "Weight hash mismatch; do not run inference"
}

Write-Host "Deployment files copied successfully."
