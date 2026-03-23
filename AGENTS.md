# Repository Guidelines

## Project Structure & Module Organization
Work from the repository root. Core code lives in `src/`: `dataset/` loads View of Delft data, `model/` contains the CenterPoint detector stack, `ops/` holds custom CUDA/C++ kernels, `config/` stores Hydra YAMLs, and `tools/` contains entry-point scripts for train/eval/test/submission. Research notes live in `experiment_plan.md`, `report.md`, and `final_assignment.ipynb`. The dataset is expected at `data/view_of_delft/` but is not committed.

## Build, Test, and Development Commands
Build native ops once before training:
```bash
cd src/ops/cpp_pkgs && python setup.py develop
```
Train locally with Hydra overrides:
```bash
python src/tools/train.py exp_id=centerpoint_radar_baseline batch_size=4 num_workers=2 epochs=12
```
Evaluate a checkpoint on validation:
```bash
python src/tools/eval.py checkpoint_path=PATH_TO_CKPT
```
Run test-set inference and package predictions:
```bash
python src/tools/test.py checkpoint_path=PATH_TO_CKPT
python src/tools/zip_files.py --res_folder outputs/EXP_ID/test_preds --output_path outputs/EXP_ID/submission.zip
```
For DelftBlue jobs, use `bash src/tools/slurm_train.sh`.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, `snake_case` for functions/variables, `PascalCase` for classes, and small modules grouped by responsibility. Keep experiment changes configurable through Hydra instead of hard-coding paths or hyperparameters. Match existing config naming such as `centerpoint_radar.yaml` and keep output folders under `outputs/<exp_id>/`.

## Testing Guidelines
There is no committed `tests/` suite yet, so treat validation runs as the minimum regression check. Before opening a PR, rebuild `src/ops/cpp_pkgs`, run the affected script, and record the command, split, and key metric changes. If you add automated tests, place them under `tests/` and name files `test_<module>.py`.

## Commit & Pull Request Guidelines
Recent history uses short imperative subjects with prefixes such as `feat:`, `perf:`, `improve:`, and `correct`. Keep commits focused and mention the subsystem changed, for example `feat: add radar temporal stacking`. PRs should state the motivation, Hydra overrides used, metrics before/after, affected configs, and any output artifacts needed for review.

## Data & Experiment Rules
Do not introduce LiDAR or stereo inputs into final methods. Preserve official train/val/test splits, keep dataset paths out of commits, and avoid adding dependencies beyond the existing `amp` environment.
