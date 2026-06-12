# poker44-innovative-dual-branch

Public release repository for Poker44 subnet (SN126) UID 164.

## Model

- **Name:** `poker44-innovative-dual-branch`
- **Version:** `reference-dualbranch-v2`
- **Framework:** LightGBM + XGBoost dual-branch stack with bounded live calibration
- **Artifact:** `models/bot_detector_v1.joblib`

## Training data

Trained only on public Poker44 benchmark releases from
`https://api.poker44.net/api/v1/benchmark` using miner-visible hand payloads.
No validator-private live labels were used.

## Implementation files

Manifest `implementation_sha256` is computed from:

- `neurons/miner.py`
- `poker44_ml/features.py`
- `poker44_ml/inference.py`
- `poker44_ml/innovative_model.py`
- `poker44_ml/stacked.py`
- `poker44_ml/rank_stack.py`
- `poker44_ml/calibration.py`

## Quick start

```bash
git clone https://github.com/Yurii214/poker-44-miner.git
cd poker-44-miner
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python verify.py
```

## Verify manifest

```bash
python verify.py
python recompute_hash.py
```

`verify.py` checks that the pinned `repo_commit` and `implementation_sha256`
in `models/model_manifest.json` match the files in this repository.

## Run miner

```bash
python neurons/miner.py --netuid 126 --wallet.name <cold> --wallet.hotkey <hot> \
  --subtensor.network finney --axon.port 8093
```
