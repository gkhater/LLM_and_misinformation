param(
    [string]$BaseConfig = "config/base.yaml",
    [string]$ModelConfig = "config/model_llama70b_groq.yaml",
    [int]$MaxRows = 0
)

.\\.venv\\Scripts\\activate

$argsList = @("-m", "src.cli", "--base-config", $BaseConfig, "--model-config", $ModelConfig)
if ($MaxRows -gt 0) { $argsList += @("--max-rows", $MaxRows) }

python @argsList
