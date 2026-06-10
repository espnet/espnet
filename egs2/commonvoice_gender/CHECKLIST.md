# PR Checklist for CommonVoice Gender Recipe

## ✅ Completed Requirements

### Recipe Structure
- [x] Recipe name follows conventions (`commonvoice_gender`)
- [x] Common/shared files are symbolic links:
  - `asr.sh` → `../../TEMPLATE/asr1/asr.sh`
  - `path.sh` → `../../TEMPLATE/asr1/path.sh`
  - `pyscripts` → `../../TEMPLATE/asr1/pyscripts`
  - `scripts` → `../../TEMPLATE/asr1/scripts`
  - `steps` → `../../TEMPLATE/asr1/steps`
  - `utils` → `../../TEMPLATE/asr1/utils`
- [x] Cluster settings are default (`cmd.sh` uses `cmd_backend='local'`)
- [x] Config files follow naming: `conf/tuning/train_asr_conformer5.yaml` (tuning variant)

### Documentation
- [x] `RESULTS.md` created with experimental results
- [x] `egs2/README.md` updated with recipe entry
- [x] `db.sh` contains recipe-specific corpus path (not modifying TEMPLATE)

### Code Quality
- [x] Recipe files: ~24 files (within limits)
- [x] Lines of code: ~867 lines (under 1000 limit)
- [x] `.gitignore` created to exclude logs, data/, exp/, dump/
- [x] No modification to TEMPLATE files (db.sh reverted)

### Experimental Results
- [x] Results documented in RESULTS.md
- [x] Same-gender and cross-gender evaluations included
- [x] Bias analysis documented

## Files to Commit

### New Recipe Directory:
```
egs2/commonvoice_gender/asr1/
├── run.sh
├── run_female.sh
├── db.sh (local file, not symlink)
├── cmd.sh
├── local/
│   ├── data.sh
│   ├── data_prep_gender.py
│   └── path.sh
├── conf/
│   ├── decode_asr.yaml
│   ├── train_lm.yaml
│   ├── tuning/
│   │   └── train_asr_conformer5.yaml
│   └── *.conf files
├── RESULTS.md
├── .gitignore
└── [helper scripts for cross-gender eval]
```

### Updated Files:
- `egs2/README.md` (added commonvoice_gender entry)

## Next Steps

1. Review files: `git status`
2. Stage recipe: `git add egs2/commonvoice_gender/ egs2/README.md`
3. Commit: `git commit -m "Add CommonVoice gender-based ASR fairness recipe"`
4. Push to fork and create PR on GitHub
5. Use PR template from `.github/PULL_REQUEST_TEMPLATE.md`
