# ICE Runbook — running LoRA fine-tuning on GT's cluster

First time using ICE? Follow these steps in order. Commands you type are in
code blocks. Anything in `<angle brackets>` you replace with your own value.

Key idea: you **log in** to a login node, **submit a job** to Slurm, and Slurm
runs your training on a **GPU compute node**. You never train on the login node.

> You (not the agent) must do the login steps — they need your GT password + Duo.

---

## 1. Log in to ICE

```bash
ssh <your_gt_username>@login-ice.pace.gatech.edu
```
Enter your GT password, then approve the Duo 2FA push on your phone.
You're now on a **login node** (fine for editing/submitting, NOT for training).

## 2. Get the code

```bash
cd ~
git clone https://github.com/kevin1285/CS4644-DeepGuessr.git
cd CS4644-DeepGuessr
```

## 3. One-time environment setup

ICE uses `module` to expose software. First find the anaconda module name:
```bash
module avail anaconda
```
If it's not `anaconda3`, edit the two `module load` lines in
`scripts/setup_ice_env.sh` and `scripts/train_lora.sbatch` to match.

Then run the setup (takes several minutes):
```bash
bash scripts/setup_ice_env.sh
```
At the end it should print `cuda available: True` — but note that's checked on
the LOGIN node, which may not have a GPU, so it's OK if it says False here.
The real GPU check happens in the job (step 6).

## 4. Get the data onto ICE

The training reads images, so the dataset must be on ICE (~8+ GB downloaded).

> **Activate the env in every new session first.** `conda` is only available
> after loading the module. If you see `conda: command not found`, run:
> ```bash
> module load anaconda3/2023.03
> conda activate deepguessr
> ```

### 4a. Point the data dir at SCRATCH before downloading (REQUIRED)
Your home directory (`/home/hice1/<user>`) has a small quota (~10-15 GB) and the
dataset will overflow it (`OSError: [Errno 122] Disk quota exceeded`). Put the
data in **scratch** first. From the repo root:
```bash
# confirm your scratch path exists (usually ~/scratch)
ls -ld ~/scratch

rm -rf deepguessr/data                      # remove any partial/overflowed download
mkdir -p ~/scratch/dg_data
ln -s ~/scratch/dg_data deepguessr/data     # symlink data/ -> scratch
ls -l deepguessr/data                       # should show '-> .../scratch/dg_data'
```
If `~/scratch` doesn't exist, find it with `ls ~ | grep -i scratch` and adjust.

### 4b. Clear any expired HuggingFace token (REQUIRED if you hit a 401)
The OSV-5M dataset is **public** and needs no token. But a leftover/expired token
in your account causes `401 Unauthorized` / `Repository Not Found`. If you see
that (look for a line like `User Access Token "..." is expired`), clear it:
```bash
hf auth logout        # or: huggingface-cli logout
rm -f ~/.cache/huggingface/token
```
Do NOT log in again — anonymous access is correct for this public dataset.

### 4c. Download + build
```bash
python deepguessr/src/download.py        # downloads + unzips shards into scratch
python deepguessr/src/build_subsets.py   # builds train/val/test CSVs
python deepguessr/src/geocells.py        # builds K=89 geocells + assignments
```
The download prints "unauthenticated requests" warnings — that's expected and
fine. It is resumable; if interrupted, just re-run it.

**Alternative — copy prebuilt CSVs/geocells from your laptop** (still need images
on ICE, so this only saves the build steps). From your LAPTOP terminal:
```bash
scp deepguessr/data/*.csv deepguessr/data/geocells_default.json \
    deepguessr/data/cell_assign_*.npy \
    <your_gt_username>@login-ice.pace.gatech.edu:~/scratch/dg_data/
```

## 5. Check what GPU partition you can use

```bash
sinfo -o "%P %G %D %t"      # lists partitions and GPU types
```
If you see a partition name you must specify (e.g. `ice-gpu`), add this line to
`scripts/train_lora.sbatch` near the other `#SBATCH` lines:
```
#SBATCH --partition=<partition_name>
```

## 6. Submit the training job

```bash
sbatch scripts/train_lora.sbatch
```
This prints a job id, e.g. `Submitted batch job 1234567`.

Monitor it:
```bash
squeue --me                       # PD = pending (queued), R = running
tail -f logs/dg_lora_<jobid>.out  # live training output (Ctrl-C to stop watching)
```
The job first prints `nvidia-smi` (confirms it got a GPU) and the throughput
(img/s) after epoch 0 — if img/s is low, you can cancel and lower
`--train-images` / `--epochs` in the sbatch file, then resubmit.

Cancel a job if needed:
```bash
scancel <jobid>
```

## 7. If the job times out or gets preempted

The script checkpoints every epoch to
`deepguessr/results/lora_geocell_smooth_tau75_ckpt.pt`. Resume by editing the
`python ...` line in the sbatch to add:
```
--resume deepguessr/results/lora_geocell_smooth_tau75_ckpt.pt
```
then `sbatch` again.

## 8. Get the results back to your laptop

When the job finishes, from your LAPTOP terminal:
```bash
scp <your_gt_username>@login-ice.pace.gatech.edu:~/CS4644-DeepGuessr/deepguessr/results/lora_geocell_* \
    deepguessr/results/
```
Then regenerate the table + figures locally:
```bash
.venv/bin/python deepguessr/src/make_results_table.py
.venv/bin/python deepguessr/src/make_figures.py
```

---

## Quick reference

| Task | Command |
|---|---|
| Log in | `ssh <user>@login-ice.pace.gatech.edu` |
| Submit job | `sbatch scripts/train_lora.sbatch` |
| My jobs | `squeue --me` |
| Watch output | `tail -f logs/dg_lora_<jobid>.out` |
| Cancel | `scancel <jobid>` |
| Partitions/GPUs | `sinfo -o "%P %G"` |

## Time budget
One LoRA run should take ≤ ~6 hours (the `--time=06:00:00` limit). After you see
the img/s throughput in the log, pick `--train-images` × `--epochs` to fit.
Handoff rule: if throughput is low, cut to 100k images × 3 epochs first.

---

## Troubleshooting (real issues seen)

**`ssh ... Operation timed out`** — you're off the GT network. Connect the GT
VPN (GlobalProtect, `vpn.gatech.edu`) first, then ssh again.

**`git clone` → `Authentication failed` / `Password authentication is not
supported`** — if the repo is private, use a GitHub Personal Access Token (not
your password) as the password. If public, the clone needs no auth.

**`conda: command not found`** — new session hasn't loaded the module. Run
`module load anaconda3/2023.03 && conda activate deepguessr`. Needed every
session (the Slurm script does this itself).

**`cuda available: False` during setup** — expected on the login node (no GPU).
CUDA is only available inside the Slurm job on a compute node.

**`401 Unauthorized` / `Repository Not Found` on download** — an expired HF token
in your account. Clear it: `hf auth logout` and
`rm -f ~/.cache/huggingface/token`. The dataset is public; don't log back in.

**`OSError: [Errno 122] Disk quota exceeded`** — home directory is full. The data
must live in scratch. Do step 4a (symlink `deepguessr/data` -> `~/scratch/dg_data`)
and re-run the download.

**`sbatch` job stuck pending (PD) forever** — you may be missing a required
`--partition`/`--account` line. Check `sinfo -o "%P %G"` and add the partition to
`scripts/train_lora.sbatch`.

## If ICE is painful (queue waits, module issues) near the deadline
Fall back to Colab Pro — ask the agent for the Colab notebook. You only need ONE
good LoRA run for full credit (E5 + the frozen-vs-LoRA comparison E6).
