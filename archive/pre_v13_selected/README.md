# Pre-V13 selected submission freeze

`submit.zip` is intentionally ignored by Git because it is approximately 892 MiB. It is the exact validated TEST5/V7 submission archive present before V13 work.

Tracked metadata:

- `artifact_manifest.json`: selected checkpoint, runtime source, fusion, Git, and ZIP hashes.
- `submission_interface.json`: immutable DACON input/output contract.

Regenerate or verify the freeze with:

```powershell
python scripts/freeze_pre_v13_submission.py
```

The V13 development process must not overwrite selected artifacts until an adoption gate passes. Existing final holdouts are forbidden for V13 selection.
