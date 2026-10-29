# DSSIM + PCC workflow

This directory keeps the DSSIM+PCC experiment separate from the existing KS
and SPRE experiments. The generated configuration files differ from their
source files only in this field:

```json
"Metric": ["dssim", "pcc"]
```

The labels require both `dssim > 0.995` and `pcc > 0.9995`. Existing feature
and metric files are read from `../data/rf_inputs_time2000`; they are not
copied or recomputed.

## Submit the complete workflow

Run this from Derecho:

```bash
cd /glade/derecho/scratch/apinard/lcr/lcr/paper_submit/dssim_pcc
./run_pipeline.sh
```

The launcher performs these operations in order using PBS dependencies:

1. Recreates `configs/` from the existing single- and multi-variable configs.
2. Regenerates both 2,000- and 1,600-timestep labels in
   `precomputed_labels/`.
3. Submits the Casper modeling array, throttled to eight simultaneous jobs.
4. Submits plotting after the modeling array finishes.

Every invocation writes to a clean timestamped directory under `runs/`. The
latest path is also written to `latest_run.txt`. Each run contains:

```text
runs/YYYY-MM-DD_HHMMSS/
  modeling_configs.txt
  run_manifest.txt
  logs/
  model_results/
  plots_single/
  plots_multi/
  label_tables/
```

The modeling step itself creates decision trees and confusion-matrix figures.
The plotting step additionally creates the single- and multi-variable F1
comparison figures, RF feature-importance comparisons, single-variable
stacked-bar products, and label-distribution CSV tables.

## Regenerate plots for an existing run

After activating `my-npl-2023a`, run:

```bash
run_dir=$(cat latest_run.txt)
./generate_plots.sh "$run_dir"
```

To submit that plotting step to Casper instead:

```bash
run_dir=$(cat latest_run.txt)
qsub -v "DSSIM_PCC_DIR=$PWD,DSSIM_PCC_RUN_DIR=$run_dir" generate_plots.pbs
```
