param(
    [string]$Grid = "fine",
    [int]$M = 80,
    [string]$Split = "all",
    [int]$StartIdx = 0,
    [int]$EndIdx = 9999,
    [switch]$KeepD
)
# SLD-ML Benchmark v1 pipeline: WSL GPU assembly -> Windows exact eigenvalues.
# Intermediate D matrices are large (fine m80: ~1.1 GB/case), so the pipeline
# processes one case at a time and deletes D after computing rho.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$DDir = Join-Path $Root "benchmark\data\D"
$RhoDir = Join-Path $Root "benchmark\data\rho"
New-Item -ItemType Directory -Force -Path $DDir, $RhoDir | Out-Null

$meta = Get-Content (Join-Path $Root "benchmark\meta.json") -Raw -Encoding UTF8 | ConvertFrom-Json
$cases = @($meta.cases)
$allCases = @($meta.cases)
if ($Split -ne "all") {
    $keys = @($meta.split.$Split | ForEach-Object { "$($_.aD)|$($_.zeta)|$($_.fn)" })
    $cases = @($cases | Where-Object { $keys -contains "$($_.aD)|$($_.zeta)|$($_.fn)" })
}
$grids = $meta.grids
$g = if ($Grid -eq "fine" -and $M -eq 80) { $grids.fine }
       elseif ($Grid -eq "fine" -and $M -eq 160) { $grids.fine160 }
       else { $grids.$Grid }
$chunk = if ($M -ge 160) { 32 } else { 128 }

Write-Host "Pipeline: grid=$Grid m=$M split=$Split cases=$($cases.Count) chunk=$chunk"
$t0 = Get-Date
for ($i = 0; $i -lt $cases.Count; $i++) {
    $case = $cases[$i]
    $idx = -1
    for ($k = 0; $k -lt $allCases.Count; $k++) {
        if ("$($allCases[$k].aD)|$($allCases[$k].zeta)|$($allCases[$k].fn)" -eq
            "$($case.aD)|$($case.zeta)|$($case.fn)") {
            $idx = $k
            break
        }
    }
    if ($idx -lt $StartIdx -or $idx -gt $EndIdx) { continue }
    $base = "case_{0:000}_{1}_m{2}" -f $idx, $Grid, $M
    $dPath = Join-Path $DDir "$base.npz"
    $rhoPath = Join-Path $RhoDir "$base.npz"
    if (Test-Path $rhoPath) {
        Write-Host "[$idx] rho exists, skip"
        continue
    }
    $t1 = Get-Date
    Write-Host "[$idx] GPU assemble $($case.aD)/$($case.zeta)/$($case.fn) ..." -NoNewline
    $wslPath = "/mnt/c/Users/tan83/Desktop/codex/ai-searching/ai-search/$($dPath.Substring($Root.Length + 1) -replace '\\','/')"
    & wsl -e bash -lc "cd /mnt/c/Users/tan83/Desktop/codex/ai-searching/ai-search && /home/tan83/yolo_env/bin/python code/sdm_solver_torch.py --aD $($case.aD) --zeta $($case.zeta) --fn $($case.fn) --m $M --grid $Grid --dump-D --out $wslPath"
    if ($LASTEXITCODE -ne 0) { throw "WSL assembly failed for case $idx" }
    Write-Host " ok ($([math]::Round(((Get-Date)-$t1).TotalSeconds,0))s)"

    $t2 = Get-Date
    Write-Host "[$idx] CPU exact eigvals ..." -NoNewline
    & python (Join-Path $Root "code\benchmark_eig_cpu.py") $dPath --out $rhoPath --nproc 16
    if ($LASTEXITCODE -ne 0) { throw "eigvals failed for case $idx" }
    Write-Host " ok ($([math]::Round(((Get-Date)-$t2).TotalSeconds,0))s)"

    if (-not $KeepD) {
        Remove-Item -LiteralPath $dPath -Force
    }
    $el = [math]::Round(((Get-Date)-$t0).TotalMinutes,1)
    Write-Host "[$idx] done. total elapsed ${el} min"
}
Write-Host "PIPELINE DONE in $([math]::Round(((Get-Date)-$t0).TotalMinutes,1)) min"
