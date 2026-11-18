# US-LesionAssist: U-Net v2 upgrade

Copy `train_v2.py` and `compare_v2.py` **into the same directory as** your existing `train.py`, `dataset.py`, `metrics.py`, and `outputs/` folder.

### From the active VS Code venv and correct project directory

```powershell
# Step 1. Check you are in the directory containing the original scripts.
Test-Path .\train.py
Test-Path .\outputs\splits.json
Test-Path .\outputs\best_model.pt

# Step 2. Train on the *same split* as v1; v1 will not be overwritten.
python train_v2.py --epochs 30 --img-size 256 --batch-size 4

# Step 3. Compare against v1 on validation data FIRST.
python compare_v2.py --split val

# Step 4. Only after deciding the model configuration, final test comparison:
python compare_v2.py --split test
```

On a GPU memory error, use `--batch-size 2` instead. If the user previously ran `app_v2.py`, it will continue loading **v1** from `outputs/best_model.pt`; the new checkpoint is saved as `outputs_v2/best_model.pt` and can be inspected with a later comparison UI.

### Changes in this experiment

- Increase input resolution from 160×160 to 256×256.
- Apply spatial flips and 90° rotations **to image and mask together** on training samples only.
- Apply modest brightness/contrast jitter and Gaussian noise to the **training image only**.
- Keep original binary segmentation BCE + soft Dice loss and U-Net architecture.
- Reuse the original `outputs/splits.json` exactly. Its SHA-256 is stored in the v2 checkpoint. Validation and test samples never receive random augmentation.
- Save v2 files in a separate folder. Do not overwrite the v1 checkpoint.
- Compare Dice/IoU both at each model's native resolution and at a shared 256×256 evaluation resolution. The latter is a fairer direct comparison, while native v1 numbers correspond to the existing baseline report.

### Caveats

This is research-only lesion segmentation. It does not predict malignancy and is not a diagnostic tool. The public BUSI data do not provide reliable patient IDs, so even exact-duplicate grouping cannot guarantee patient-independent evaluation. The project has already viewed test results for v1: treat this as an exploratory benchmark and **not** a pristine, previously untouched test set. A truly independent test needs an external cohort or patient-identified data.

A higher validation score is not guaranteed. `malignant (106).png` is a validation image and is a useful illustrative failure case, but do not optimize exclusively for that one image; compare aggregate results too.
