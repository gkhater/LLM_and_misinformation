# Data layout

- `claims.csv` columns:
  - `id`: unique integer identifier.
  - `split`: use to filter runs (e.g., `eval`, `test`). Controlled via `config/base.yaml: dataset.split_filter`.
  - `claim`: claim text to classify.
  - `context`: optional supporting/contrasting context (can be blank).

To run on all rows, set `dataset.split_filter` to `null` in `config/base.yaml`. Replace the provided examples with your evaluation set as needed.
