# Results

Generated metrics, figures, and logs are written here.

The baseline demo writes a JSON metrics file to `results/metrics/` and a trajectory plot to `results/figures/`.

The baseline matrix writes one JSON metrics file per synthetic scenario plus `results/metrics/baseline_matrix_summary.csv`.

The closed-loop baseline writes per-step CSV logs to `results/logs/`, per-scenario JSON metrics to `results/metrics/`, and `results/metrics/closed_loop_baseline_summary.csv`.

The planner comparison writes per-planner/per-scenario logs and metrics, plus `results/metrics/planner_comparison_summary.csv`.

The synthetic uncertainty comparison writes per-planner logs and metrics for `ambiguous_cut_in` and `delayed_cut_in`, plus `results/metrics/uncertainty_planner_comparison_summary.csv`.

The analysis command writes thesis-ready tables to `results/tables/`, plots to `results/figures/`, and an analysis manifest to `results/analysis/`. The main polished table is `results/tables/key_findings_table.csv`.
