# ReObsDiff implementation

The original GTCom entry points remain untouched. ReObsDiff is isolated under
`reobsdiff/`, `configs/reobsdiff.yaml`, and `scripts/*reobsdiff.py`.

## Typical workflow

```bash
python scripts/build_reobs_cache.py --scene /path/to/session --output cache/session \
  --config configs/reobsdiff.yaml --resume
accelerate launch scripts/train_reobsdiff.py --cache cache/session \
  --config configs/reobsdiff.yaml --output models/reobsdiff-lora --steps 5000
python scripts/infer_reobsdiff.py --input output/scene/warps --output output/scene/reobsdiff \
  --lora models/reobsdiff-lora
python scripts/audit_target_leakage.py --cache cache/session \
  --training-output models/reobsdiff-lora
```

For the first GT-depth smoke run, add `--set depth_source=gt`. Published iMED
Task-2 scenes do not expose E2-right depth/extrinsics; stereo supervision is
therefore used only for datasets/caches that provide both. Temporal E2-L
re-observation remains active. The FoundationStereo provider consumes metric
depth caches and intentionally refuses to run a model during training.
