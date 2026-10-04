$ErrorActionPreference = "Continue"

$base_dir = $PSScriptRoot
$logs_dir = Join-Path $base_dir "logs"
if (-Not (Test-Path $logs_dir)) {
    New-Item -ItemType Directory -Force -Path $logs_dir | Out-Null
}

$deps = "--with pandas --with numpy --with matplotlib --with seaborn --with scikit-learn --with torch"

$scripts = @(
    "left\MLP\train_mlp_pls.py",
    "left\SVM\train_svm_pls.py",
    "left\ResNet\train_resnet_pls.py",
    "left\ResNet\train_resnet_ae_pls.py",
    "left\MobileNet\train_mobilenet_pls.py",
    "right\MLP\train_mlp_pls.py",
    "right\SVM\train_svm_pls.py",
    "right\ResNet\train_resnet_pls.py",
    "right\ResNet\train_resnet_ae_pls.py",
    "right\MobileNet\train_mobilenet_pls.py"
)

Set-Location $base_dir

foreach ($script in $scripts) {
    Write-Host "Running $script ..."
    
    $name = $script -replace "\\", "_"
    $name = $name -replace "\.py", ""
    $log_file = Join-Path $logs_dir "$name.log"
    
    $script_dir = Split-Path $script
    $script_name = Split-Path $script -Leaf
    
    Push-Location $script_dir
    
    $cmd = "uv run $deps $script_name *>&1 | Out-File -FilePath '$log_file' -Encoding UTF8"
    Invoke-Expression $cmd
    
    Pop-Location
    Write-Host "Completed $script. Log saved to $log_file"
}

Write-Host "All models finished running."
