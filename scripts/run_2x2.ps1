# Usage:
#   .\scripts\run_2x2.ps1 -ClaimsCsv data\demo_claims_demo12_balanced.csv -EvalConfig config\eval_demo_balanced_853.yaml -Model8 config\model_llama8b_groq_liar.yaml -Model70 config\model_llama70b_groq_liar.yaml -OutPrefix liar_demo
param(
    [string]$ClaimsCsv,
    [string]$EvalConfig,
    [string]$Model8,
    [string]$Model70,
    [string]$OutPrefix = "run"
)

if (-not $ClaimsCsv -or -not $EvalConfig -or -not $Model8 -or -not $Model70) {
    Write-Error "Usage: run_2x2.ps1 -ClaimsCsv <csv> -EvalConfig <yaml> -Model8 <yaml> -Model70 <yaml> [-OutPrefix prefix]"
    exit 1
}

$gen8 = "outputs/gen_${OutPrefix}_8b.jsonl"
$gen70 = "outputs/gen_${OutPrefix}_70b.jsonl"
$eval8 = "outputs/eval_${OutPrefix}_8b.jsonl"
$eval70 = "outputs/eval_${OutPrefix}_70b.jsonl"
$rep8 = "outputs/report_${OutPrefix}_8b.md"
$rep70 = "outputs/report_${OutPrefix}_70b.md"
$cmp = "outputs/compare_${OutPrefix}.md"

Write-Host "Generation 8B -> $gen8"
python -m src.cli --mode generation --model-config $Model8 --claims-csv $ClaimsCsv --max-rows 0 --base-config config/base.yaml
$latestGen8 = Get-ChildItem outputs\gen_*_*.jsonl | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $latestGen8) { Write-Error "No gen output for 8B"; exit 1 }
Rename-Item $latestGen8.FullName $gen8 -Force

Write-Host "Generation 70B -> $gen70"
python -m src.cli --mode generation --model-config $Model70 --claims-csv $ClaimsCsv --max-rows 0 --base-config config/base.yaml
$latestGen70 = Get-ChildItem outputs\gen_*_*.jsonl | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $latestGen70) { Write-Error "No gen output for 70B"; exit 1 }
Rename-Item $latestGen70.FullName $gen70 -Force

Write-Host "Evaluation 8B -> $eval8"
python -m src.cli --mode evaluation --eval-config $EvalConfig --input-jsonl $gen8 --claims-csv "" --max-rows 0 --smoke 0
$latestEval8 = Get-ChildItem outputs\eval_*_*.jsonl | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $latestEval8) { Write-Error "No eval output for 8B"; exit 1 }
Rename-Item $latestEval8.FullName $eval8 -Force

Write-Host "Evaluation 70B -> $eval70"
python -m src.cli --mode evaluation --eval-config $EvalConfig --input-jsonl $gen70 --claims-csv "" --max-rows 0 --smoke 0
$latestEval70 = Get-ChildItem outputs\eval_*_*.jsonl | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if (-not $latestEval70) { Write-Error "No eval output for 70B"; exit 1 }
Rename-Item $latestEval70.FullName $eval70 -Force

Write-Host "Reports -> $rep8 / $rep70"
python scripts/print_demo_report.py --eval-jsonl $eval8 --out-md $rep8
python scripts/print_demo_report.py --eval-jsonl $eval70 --out-md $rep70

Write-Host "Comparison -> $cmp"
python scripts/compare_two_models.py --eval8 $eval8 --eval70 $eval70 --out $cmp

Write-Host "Done. Files:"
Write-Host "  $gen8"
Write-Host "  $gen70"
Write-Host "  $eval8"
Write-Host "  $eval70"
Write-Host "  $rep8"
Write-Host "  $rep70"
Write-Host "  $cmp"
