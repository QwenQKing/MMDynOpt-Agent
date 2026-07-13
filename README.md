# MMDynOpt

## Environment

- Python 3.12+
- CUDA 12.4+ (match your PyTorch / vLLM build)

```bash
pip install verl>=0.4.0
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

Set `PYTHONPATH` so the package is importable (from the `MMDynOpt` directory):

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"
```

Configure the external multimodal LLM in `mmdynopt_agent/utils/tools/mm_LM_env.py` (`BASE_URL`, `MODEL`, `API_KEY`).

## Training (quick start)

```bash
cd MMDynOpt
# Edit mmdynopt_agent/scripts/run.sh: BASE_MODEL, TRAIN_DATA_PATH, VAL_DATA_PATH, GPUs, Ray ports as needed.
bash mmdynopt_agent/scripts/run.sh
```

## Testing / evaluation (quick start)

```bash
cd MMDynOpt
export PYTHONPATH="$(pwd):${PYTHONPATH}"
# Edit mmdynopt_agent/scripts/eval.sh: MODEL_PATH, CHECKPOINT_PATH, VAL_DATA_DIR, RESULTS_BASE_DIR, GPUS, Ray ports as needed.
bash mmdynopt_agent/scripts/eval.sh
```

Optional: aggregate metrics from saved `res.json` trees:

```bash
python -m mmdynopt_agent.scripts.eval --res-dir ./your_results_dir
```
