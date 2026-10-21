#!/bin/bash
#SBATCH --job-name=lrat-infoseek-judge
#SBATCH --partition=h100sxm5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --chdir=/scratch.hpc/mehdi.kabbage/lrat-reproduction
#SBATCH --output=judge-infoseek-%j.log

# Usage: sbatch judge_infoseek.sbatch baseline-v1
# Scores both conditions with the original judging protocol and a recorded
# same-node communication workaround. Test communication before loading weights.
set -euo pipefail
tag="${1:-baseline-v1}"
[[ "$tag" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid experiment tag."; exit 1; }
project="$PWD"
root="$project/runs/infoseek-20-$tag"
image="$project/containers/pytorch-2.7.0-cu126-devel.sif"
judge="$project/models/Qwen3-30B-A3B-Thinking-2507-144afc2f"
output="$root/judge-localcomm-job-$SLURM_JOB_ID"
[[ -d "$root" ]] || { echo "Missing experiment: $root"; exit 1; }
[[ -s "$image" ]] || { echo "Missing container: $image"; exit 1; }
: "${CUDA_VISIBLE_DEVICES:?Slurm did not expose the allocated GPUs.}"
exec 9>"$root/.judge.lock"
flock -n 9 || { echo "A judge job is already running for this experiment."; exit 1; }
mkdir "$output"

echo "Judge node: $(hostname); job: $SLURM_JOB_ID"
echo "Two GPUs share the judge model. No agent generation or corpus encoding."
echo "Evaluating all 20 saved outcomes for each retriever, including nonanswers."
echo "Results: $output"
echo "Runtime workaround: localhost networking; custom all-reduce disabled."

container_python() {
    timeout --signal=TERM --kill-after=15s "${judge_command_timeout:-2700}s" \
      apptainer exec --cleanenv --nv \
      --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
      --env HF_HOME="$project/hf-cache" --env HF_HUB_OFFLINE=1 \
      --env VLLM_CACHE_ROOT="$project/vllm-cache" \
      --env VLLM_WORKER_MULTIPROC_METHOD=spawn \
      --env VLLM_HOST_IP=127.0.0.1 \
      --env GLOO_SOCKET_IFNAME=lo --env NCCL_SOCKET_IFNAME=lo \
      --env NCCL_IB_DISABLE=1 --env NCCL_CUMEM_ENABLE=0 --env NCCL_DEBUG=INFO \
      --env OMP_NUM_THREADS=4 --env OPENBLAS_NUM_THREADS=1 \
      --bind /scratch.hpc/mehdi.kabbage:/scratch.hpc/mehdi.kabbage \
      --pwd "$project" "$image" "$project/venv-vllm/bin/python" -u "$@"
}

# This uses the same allocated GPUs, without loading the 61 GB model.
# Each collective has a short timeout; the outer timeout also covers CUDA hangs.
cat > "$output/communication-test.py" <<'COMMUNICATION'
import json
import os
import socket
import sys
from datetime import timedelta
from pathlib import Path

def worker(rank):
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", rank=rank, world_size=2, timeout=timedelta(seconds=90))
    print(f"Communication test: rank {rank} Gloo initialized.", flush=True)
    value = torch.tensor([float(rank + 1)])
    dist.all_reduce(value)
    assert value.item() == 3.0, "Gloo reduction failed."
    gpu_group = dist.new_group(ranks=[0, 1], backend="nccl", timeout=timedelta(seconds=90))
    value = torch.tensor([float(rank + 1)], device=f"cuda:{rank}")
    dist.all_reduce(value, group=gpu_group)
    torch.cuda.synchronize()
    assert value.item() == 3.0, "PyTorch NCCL reduction failed."
    print(f"Communication test: rank {rank} PyTorch NCCL passed.", flush=True)
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    communicator = PyNcclCommunicator(group=dist.group.WORLD, device=rank)
    value = torch.tensor([float(rank + 1)], device=f"cuda:{rank}")
    reduced = communicator.all_reduce(value)
    torch.cuda.synchronize()
    assert reduced.item() == 3.0, "vLLM NCCL reduction failed."
    print(f"Communication test: rank {rank} vLLM NCCL passed.", flush=True)
    dist.barrier()
    dist.destroy_process_group(gpu_group)
    dist.destroy_process_group()

if __name__ == "__main__":
    import torch.multiprocessing as mp
    # One node only: the local loopback address is reachable by both workers.
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        os.environ["MASTER_PORT"] = str(listener.getsockname()[1])
    mp.spawn(worker, nprocs=2, join=True)
    Path(sys.argv[1]).write_text(json.dumps({"passed": True, "ranks": 2,
        "tests": ["Gloo", "PyTorch NCCL", "vLLM PyNcclCommunicator"],
        "environment": {key: os.environ.get(key) for key in (
            "VLLM_HOST_IP", "GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME",
            "NCCL_IB_DISABLE", "NCCL_CUMEM_ENABLE")}}, indent=2) + "\n")
    print("Two-GPU communication test passed.", flush=True)
COMMUNICATION

echo "Checking Gloo and NCCL communication first (five-minute maximum)."
if ! judge_command_timeout=300 container_python "$output/communication-test.py" "$output/communication-test.json"; then
    echo "STOP: communication test failed or timed out. Judge weights were not loaded."
    exit 1
fi

# Freeze exactly the 40 original-format inputs and the evaluator before loading GPUs.
container_python - "$project" "$root" "$judge" "$output" <<'PREFLIGHT'
import csv, hashlib, json, math, os, shutil, sys
from pathlib import Path
import torch, vllm

project, root, judge, out = map(Path, sys.argv[1:])
expected_source = "df4b1d0cd7f6e20bd492ee4c224d46fa6e079e3f51fbeab0692cffb122e71942"
expected_revision = "144afc2f379b542fdd4e85a1fcd5e1f79112d95d"
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def require(condition, message):
    if not condition:
        raise ValueError(message)
require(torch.cuda.device_count() == 2, "Expected exactly two allocated GPUs.")
require(vllm.__version__.split("+")[0] == "0.9.0.1", "Use the existing vLLM 0.9.0.1 environment.")
source = project.parent / "LRAT/scripts_evaluation/evaluate.py"
require(sha(source) == expected_source, "The pinned original evaluator changed.")
provenance = json.loads((judge / "download-provenance.json").read_text())
require(provenance["model"] == "Qwen/Qwen3-30B-A3B-Thinking-2507"
        and provenance["revision"] == expected_revision, "Unexpected judge checkpoint.")
require(sha(judge / "config.json") == provenance["config_sha256"]
        and sha(judge / "model.safetensors.index.json") == provenance["weight_index_sha256"],
        "Judge configuration or weight index changed since download.")
weights = sorted(set(json.loads((judge / "model.safetensors.index.json").read_text())["weight_map"].values()))
require(len(weights) == 16 and sorted(row["file"] for row in provenance["weights"]) == weights,
        "Expected the 16 verified judge weight files.")
for row in provenance["weights"]:
    require((judge / row["file"]).stat().st_size == row["bytes"], "Missing or incomplete judge weights.")
with (project / "data/infoseek/queries.tsv").open(newline="") as stream:
    queries = list(csv.reader(stream, delimiter="\t"))
require(len(queries) == 300 and all(len(row) == 2 for row in queries), "Unexpected question file.")
selected = [[qid.strip(), question.strip()] for qid, question in queries[:20]]
require(len({qid for qid, _ in selected}) == 20, "Duplicate selected question IDs.")
gt_path = project / "data/infoseek/ground-truth.tsv"
with gt_path.open(newline="") as stream:
    rows = list(csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE))
require(len(rows) == 300 and all(len(row) >= 3 for row in rows), "Expected the official 300-row ground truth.")
truth = {row[0].strip(): (row[1].strip(), row[2].strip()) for row in rows}
require(len(truth) == 300 and all(truth[qid][0] == question and truth[qid][1]
                                for qid, question in selected), "Questions do not match ground truth.")
inputs = out / "inputs"
inputs.mkdir()
manifest = {"selected_questions": selected, "inputs": {}, "protocols": {},
            "judge": provenance, "evaluator_sha256": expected_source,
            "execution_workaround": {"disable_custom_all_reduce": True, "max_model_len": 65536,
                "environment": {key: os.environ.get(key) for key in (
                    "VLLM_HOST_IP", "GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME",
                    "NCCL_IB_DISABLE", "NCCL_CUMEM_ENABLE")},
                "communication_test": json.loads((out / "communication-test.json").read_text()),
                "command_timeout_seconds": 2700,
                "reason": "Job 59331 timed out during two-GPU distributed initialization."},
            "evaluation_settings": {"temperature": 0.0, "max_tokens": 4096, "official_max_tokens": 512, "seed": 2026,
                "tensor_parallel_size": 2, "gpu_memory_utilization": 0.9, "batch_size": 32,
                "prompt": "Original JUDGE_TEMPLATE; raw prompt, no added chat template."},
            "runtime": {"vllm": vllm.__version__, "torch": torch.__version__,
                "gpus": [torch.cuda.get_device_name(i) for i in range(2)]}}
common_protocol = None
for retriever in ("qwen", "lrat"):
    folder = root / retriever
    protocol = json.loads((folder / "protocol.json").read_text())
    identity = protocol["identity"]
    require(identity["queries"] == selected, f"{retriever}: not the same first 20 questions.")
    if common_protocol is None:
        common_protocol = identity["protocol"]
    require(identity["protocol"] == common_protocol, "Agent protocol differs between retrievers.")
    summary = json.loads((folder / "sample-summary.json").read_text())
    require(summary["retriever"] == retriever and summary["questions"] == summary["processed"] == 20
            and summary["infrastructure_errors"] == 0, f"{retriever}: incomplete run or infrastructure error.")
    require(len(summary["results"]) == 20
            and {row["query_id"] for row in summary["results"]} == {qid for qid, _ in selected},
            f"{retriever}: summary question IDs differ.")
    expected = {f"run_{qid}.json" for qid, _ in selected}
    require({p.name for p in folder.glob("run_*.json")} == expected,
            f"{retriever}: missing or unexpected trajectory files.")
    # This is the same flat-file selection that the unmodified evaluator uses.
    recognized = []
    for path in folder.glob("*.json"):
        data = json.loads(path.read_text())
        if str(data.get("query_id")) in truth:
            recognized.append(path.name)
    require(set(recognized) == expected, f"{retriever}: extra evaluator-readable files.")
    for qid, _ in selected:
        source_path = folder / f"run_{qid}.json"
        raw = json.loads(source_path.read_text())
        require(raw["query_id"] == qid, "A trajectory has the wrong question ID.")
        counts = raw.get("tool_call_counts", {}) or {}
        require(isinstance(counts, dict) and all(isinstance(v, (int, float)) and math.isfinite(v)
                and v >= 0 for v in counts.values()), "Invalid saved tool counts.")
        name = f"{retriever}__run_{qid}.json"
        shutil.copyfile(source_path, inputs / name)
        manifest["inputs"][name] = {"retriever": retriever, "query_id": qid,
            "source": str(source_path), "sha256": sha(inputs / name),
            "steps": int(sum(counts.values()))}
    manifest["protocols"][retriever] = protocol
shutil.copyfile(gt_path, out / "ground-truth.tsv")
shutil.copyfile(source, out / "official-evaluate.py")
# Only the job's private copy gets one additional, documented vLLM keyword.
# Keep the original repo file and the byte-identical snapshot unchanged.
needle = "        trust_remote_code=True,\n    )"
replacement = "        trust_remote_code=True,\n        disable_custom_all_reduce=True,\n        max_model_len=65536,\n    )"
original_code = source.read_text()
require(original_code.count(needle) == 1, "Unexpected original LLM constructor.")
local_code = original_code.replace(needle, replacement, 1)
require(local_code.count("max_tokens=512,") == 1, "Unexpected judge budget.")
local_code = local_code.replace("max_tokens=512,", "max_tokens=4096,", 1)

compile(local_code, "evaluate-local-comm.py", "exec")
(out / "evaluate-local-comm.py").write_text(local_code)
manifest["executed_evaluator_sha256"] = sha(out / "evaluate-local-comm.py")
manifest["ground_truth_sha256"] = sha(out / "ground-truth.tsv")
(out / "judge-protocol.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
print("Preflight passed: 20 Qwen + 20 LRAT outcomes; no outcomes excluded.", flush=True)
print("Original judging prompt, parser and 4096-token budget (official: 512); custom all-reduce disabled in a private copy.", flush=True)
PREFLIGHT

# Names carry the retriever label; the question IDs and raw JSON contents are unchanged.
# One invocation scores all 40 files, avoiding loading the 61 GB judge twice.
container_python "$output/evaluate-local-comm.py" \
  --input_dir "$output/inputs" \
  --gt_path "$output/ground-truth.tsv" \
  --dataset_type InfoSeek-Eval \
  --output_file "$output/official-all-results.json" \
  --model_path "$judge" \
  --tensor_parallel_size 2 --gpu_memory_utilization 0.9 --batch_size 32

# Split official per-file verdicts into the two conditions; do not rejudge or edit them.
container_python - "$output" <<'AUDIT'
import hashlib, json, math, re, sys
from pathlib import Path
out = Path(sys.argv[1])
manifest = json.loads((out / "judge-protocol.json").read_text())
original = json.loads((out / "official-all-results.json").read_text())
details = original["details"]
expected = manifest["inputs"]
files = [row["file"] for row in details]
if len(files) != 40 or len(set(files)) != 40 or set(files) != set(expected):
    raise ValueError("The official evaluator did not return exactly the 40 selected outcomes.")
for row in details:
    item = expected[row["file"]]
    if row["query_id"] != item["query_id"] or row["steps"] != item["steps"]:
        raise ValueError("An evaluated question ID or tool count differs from its saved input.")
    if hashlib.sha256((out / "inputs" / row["file"]).read_bytes()).hexdigest() != item["sha256"]:
        raise ValueError("A frozen evaluation input changed during judging.")
comparison = {"questions_per_retriever": 20, "conditions": {},
              "protocol": "Original judging prompt/parser; output budget 4096 (official: 512); recorded same-node communication workaround.",
              "execution_workaround": manifest["execution_workaround"]}
for retriever in ("qwen", "lrat"):
    rows = [row for row in details if expected[row["file"]]["retriever"] == retriever]
    if len(rows) != 20:
        raise ValueError("Incorrect per-retriever denominator.")
    metrics = {"Success Rate": sum(row["correct"] for row in rows) / 20,
               "Avg Steps": sum(row["steps"] for row in rows) / 20}
    unparsed = [row["query_id"] for row in rows
                if not re.search(r"correct:\s*(yes|no)", row.get("judge_reason", ""), re.IGNORECASE)]
    comparison["conditions"][retriever] = {"metrics": metrics, "evaluated": 20,
        "correct": sum(row["correct"] for row in rows), "unparsed_judge_verdicts": unparsed}
    (out / f"{retriever}-eval.json").write_text(
        json.dumps({"metrics": metrics, "details": rows}, indent=2, ensure_ascii=False) + "\n")
    print(f"{retriever}: {sum(row['correct'] for row in rows)}/20 correct according to the official parser; "
          f"SR {100 * metrics['Success Rate']:.1f}%; Avg Steps {metrics['Avg Steps']:.2f}; "
          f"unparsed judge verdicts: {len(unparsed)}", flush=True)
all_correct = sum(row["correct"] for row in details) / 40
all_steps = sum(row["steps"] for row in details) / 40
if not (math.isclose(original["metrics"]["Success Rate"], all_correct)
        and math.isclose(original["metrics"]["Avg Steps"], all_steps)):
    raise ValueError("Split metrics disagree with the original aggregate metrics.")
comparison["usable_for_comparison"] = not any(
    condition["unparsed_judge_verdicts"] for condition in comparison["conditions"].values())
(out / "comparison.json").write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n")
print("Comparison:", out / "comparison.json", flush=True)
if not comparison["usable_for_comparison"]:
    print("INSPECTION REQUIRED: some judge responses lack a parsable yes/no verdict. "
          "The official parser counted them as incorrect. Inspect saved judge_reason text "
          "before interpreting success rates; do not silently change the judge protocol.", flush=True)
    sys.exit(2)
print("SUCCESS: both conditions evaluated; all 40 outcomes retained and verdicts parsable.", flush=True)
AUDIT
