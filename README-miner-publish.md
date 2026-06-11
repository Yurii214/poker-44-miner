# Poker44 UID 164 miner publish

This miner advertises:

- Repo: https://github.com/Yurii214/poker-44-miner
- Model: `poker44-innovative-dual-branch`

Validators require the declared `repo_commit` to exist on that public repo with the
implementation files listed in the miner manifest.

## Publish from this server

```bash
cd /root/bittensor-mining/Poker44-subnet
export GITHUB_TOKEN="<classic PAT with repo scope>"
./scripts/publish_miner_repo.sh
```

The script pushes the miner implementation, pins the commit in
`/root/bittensor-mining/scripts/start_sn126_miner.sh`, and restarts `sn126-miner`.

## SSH deploy key alternative

Add this public key at https://github.com/Yurii214/poker-44-miner/settings/keys
with **Allow write access**:

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIE2gz+++xYvnGWrjbz94j/dSjKxDjLxTVsBG/BPkrEAY poker44-uid164-deploy
```

Then run:

```bash
./scripts/publish_miner_repo.sh
```
