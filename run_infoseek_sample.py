#!/usr/bin/env python3
"""Run a fixed InfoSeek sample with the released agent loop and resumable outputs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import time
from types import SimpleNamespace
import urllib.request

import run_infoseek_pilot as helper


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def selected_questions(project, limit):
    path = project / "data/infoseek/queries.tsv"
    full = helper.read_queries(path)
    if len(full) != 300 or not 1 <= limit <= len(full):
        raise ValueError("Expected 300 official questions and a sample size between 1 and 300.")
    selected = full[:limit]
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", qid) for qid, _ in selected):
        raise ValueError("Unexpected question-ID format.")
    return path, selected


def publish_trajectory(output, record, trajectory):
    name = record["trajectory"]
    if name != f"run_{record['query_id']}.json" or trajectory.get("query_id") != record["query_id"]:
        raise ValueError("A checkpoint contains a different question or trajectory filename.")
    destination = output / name
    if destination.exists():
        if digest(json.loads(destination.read_text())) != digest(trajectory):
            raise ValueError(f"Saved trajectory changed: {destination}")
    else:
        helper.atomic_json(destination, trajectory)


def save_question(output, identity_sha, record, trajectory):
    checkpoint = output / "checkpoints" / f"{record['query_id']}.json"
    if checkpoint.exists():
        raise ValueError(f"Refusing to replace a completed question: {checkpoint}")
    # The checkpoint atomically commits both the record and original raw output.
    # If the job ends before publication, resume restores the flat judge input.
    helper.atomic_json(checkpoint, {
        "identity_sha256": identity_sha, "record": record,
        "record_sha256": digest(record),
        "trajectory_sha256": digest(trajectory), "original_trajectory": trajectory,
    })
    publish_trajectory(output, record, trajectory)


def restore_questions(output, identity_sha, queries):
    selected = dict(queries)
    records = {}
    for checkpoint in sorted((output / "checkpoints").glob("*.json")):
        data = json.loads(checkpoint.read_text())
        row, trajectory = data["record"], data["original_trajectory"]
        qid = row["query_id"]
        if (data["identity_sha256"] != identity_sha or qid not in selected
                or row["question"] != selected[qid] or checkpoint.name != f"{qid}.json"
                or data["record_sha256"] != digest(row)
                or data["trajectory_sha256"] != digest(trajectory) or qid in records):
            raise ValueError(f"Checkpoint does not match this experiment: {checkpoint}")
        publish_trajectory(output, row, trajectory)
        if row.get("error") or row.get("retrieval_errors"):
            raise RuntimeError(f"Inspect infrastructure error in {checkpoint}; it will not be silently retried.")
        records[qid] = row
    expected = {row["trajectory"] for row in records.values()}
    unexpected = {p.name for p in output.glob("run_*.json")} - expected
    if unexpected:
        raise ValueError(f"Untracked trajectories in output directory: {sorted(unexpected)}")
    return records


def write_summary(output, retriever, queries, records):
    rows = [records[qid] for qid, _ in queries if qid in records]
    compact = [{k: v for k, v in row.items()
                if k not in ("searches", "documents_read", "request_budgets")} for row in rows]
    summary = {
        "purpose": "Generation outcomes only; correctness awaits the official judge.",
        "retriever": retriever, "questions": len(queries), "processed": len(rows),
        "answer_text_present": sum(row["completed_with_answer"] for row in rows),
        "answer_status_counts": dict(Counter(row["answer_status"] for row in rows)),
        "infrastructure_errors": sum(bool(row.get("error") or row.get("retrieval_errors")) for row in rows),
        "results": compact,
    }
    helper.atomic_json(output / "sample-summary.json", summary)
    return summary


def execute_questions(output, identity_sha, queries, records, retriever, settings,
                      make_agent, searcher, serializer, port, server_model):
    for number, (qid, question) in enumerate(queries, 1):
        if qid in records:
            print(f"Reusing saved outcome {number}/{len(queries)}: {qid}", flush=True)
            continue
        print(f"Question {number}/{len(queries)}: {qid} | {question}", flush=True)
        searcher.search_log.clear(); searcher.document_log.clear(); searcher.retrieval_errors.clear()
        agent = make_agent()
        started = time.perf_counter()
        try:
            result = agent._run({"item": {"question": question, "answer": ""},
                                 "planning_port": port}, server_model)
        except Exception as error:
            result = {"prediction": "", "termination": "request_error", "error": str(error),
                      "messages": agent.last_request_messages, "tool_call_counts": {},
                      "tool_call_counts_all": {}, "retrieved_docids": []}
        budgets = agent.request_budgets
        finish_reason = budgets[-1].get("finish_reason") if budgets else None
        assessment = helper.assess_answer(result.get("prediction", ""), result.get("termination"),
                                          result.get("error"), searcher.retrieval_errors, finish_reason)
        record = {
            "query_id": qid, "question": question, **assessment,
            "termination": result.get("termination"), "error": result.get("error"),
            "retrieval_errors": list(searcher.retrieval_errors),
            "tool_calls": result.get("tool_call_counts", {}),
            "searches": list(searcher.search_log), "documents_read": list(searcher.document_log),
            "request_budgets": list(budgets), "seconds": time.perf_counter() - started,
            "trajectory": f"run_{qid}.json", "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        }
        staging = output / "staging"
        staging.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=qid + "-", dir=staging) as temporary:
            # Preserve the repository's serializer and raw response. The answer
            # classifier affects reporting, not the text passed to the judge.
            raw_file = serializer(Path(temporary), qid, question, result, settings)
            trajectory = json.loads(raw_file.read_text())
        save_question(output, identity_sha, record, trajectory)
        records[qid] = record
        write_summary(output, retriever, queries, records)
        print("Saved:", qid, "| outcome:", record["answer_status"],
              "| tool calls:", record["tool_calls"], flush=True)
        if record["error"] or record["retrieval_errors"]:
            raise RuntimeError("Infrastructure error saved; inspect it before continuing this experiment.")
    return write_summary(output, retriever, queries, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=helper.DEFAULT_PROJECT)
    parser.add_argument("--retriever", choices=("qwen", "lrat"), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if not hasattr(helper, "assess_answer"):
        parser.error("Copy the current run_infoseek_pilot.py alongside this script first.")
    project = args.project.resolve()
    os.chdir(project)
    repo = project.parent / "LRAT"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    queries_path, queries = selected_questions(project, args.limit)
    # Lock before loading any index or model; duplicate submissions fail promptly.
    with (output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another job is already writing to this experiment directory.")
        import faiss
        import torch
        import flash_attn
        import transformers
        import datasets
        import pyarrow
        from openai import OpenAI
        if (torch.cuda.device_count() != 1 or flash_attn.__version__ != "2.8.3"
                or torch.__version__.split("+")[0] != "2.7.0"
                or transformers.__version__ != "4.53.2" or faiss.get_num_gpus() != 0):
            raise RuntimeError("Expected one GPU, the tested FA2 environment, and faiss-cpu.")
        faiss.omp_set_num_threads(int(os.getenv("OMP_NUM_THREADS", "8")))
        _, metadata = helper.corpus_info(project)
        cache = json.loads((project / "data/wiki-25/encoding-cache.json").read_text())
        if (cache["documents"] != metadata["documents"]
                or cache["corpus_sha256"] != metadata["corpus_sha256"]
                or cache["encoder_source_hashes"] != helper.source_hashes(repo)
                or cache["datasets_version"] != datasets.__version__
                or cache["pyarrow_version"] != pyarrow.__version__):
            raise ValueError("Corpus, encoder, or dataset environment differs from the completed encodings.")
        for row in cache["cache_files"]:
            if Path(row["filename"]).stat().st_size != row["bytes"]:
                raise ValueError("An encoding-cache file is missing or incomplete.")
        model = cache["models"][args.retriever]
        model_path = Path(model["path"])
        stat = (model_path / "model.safetensors").stat()
        if (helper.sha256_file(model_path / "config.json") != model["config_sha256"]
                or stat.st_size != model["weights_bytes"] or stat.st_mtime_ns != model["weights_mtime_ns"]):
            raise ValueError("The pinned retriever checkpoint changed.")
        with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=10) as stream:
            server = json.load(stream)["data"][0]
        agent_model = project / "models/AgentCPM-Explore"
        if server["id"] != str(agent_model) or server.get("max_model_len") != 131072:
            raise ValueError("Expected the same AgentCPM model with context limit 131072.")
        original_searcher, original_agent, tools = helper.load_original_modules(repo)
        index_folder = project / "indexes" / helper.RUN_NAME / args.retriever
        settings = SimpleNamespace(
            retriever=args.retriever, index_path=str(index_folder), model_name=str(model_path),
            normalize=True, pooling="eos", torch_dtype="float16", dataset_name=cache["corpus_path"],
            task_prefix="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
            max_length=8192, model=str(agent_model), temperature=0.85, top_p=0.95,
            presence_penalty=1.1, snippet_max_tokens=64, k=10, searcher_type="faiss",
        )
        identity = {
            "protocol": helper.PROTOCOL, "agent_loop": "Unmodified pinned LRAT _run; no final-answer patch.",
            "sampling_note": "Paper settings: temperature 0.85, seed 2025; pinned API code defaults differ.",
            "queries_sha256": helper.sha256_file(queries_path), "queries": queries,
            "retriever": model, "index_metadata_sha256": helper.sha256_file(index_folder / "metadata.json"),
            "corpus_sha256": cache["corpus_sha256"], "source_hashes": helper.SOURCE_HASHES,
            "sample_runner_sha256": helper.sha256_file(Path(__file__)),
            "pilot_helper_sha256": helper.sha256_file(Path(helper.__file__)),
            "server_model": server["id"], "server_context_limit": server["max_model_len"],
            "agent_config_sha256": helper.sha256_file(agent_model / "config.json"),
            "agent_generation_config": (json.loads((agent_model / "generation_config.json").read_text())
                                        if (agent_model / "generation_config.json").exists() else None),
            "agent_weight_files": [{"file": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                                   for p in sorted(agent_model.glob("*.safetensors"))],
            "runtime_versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                                 "faiss": faiss.__version__, "flash_attention": flash_attn.__version__,
                                 "datasets": datasets.__version__, "pyarrow": pyarrow.__version__},
        }
        identity_sha = digest(identity)
        protocol_path = output / "protocol.json"
        if protocol_path.exists():
            saved_protocol = json.loads(protocol_path.read_text())
            if (digest(saved_protocol["identity"]) != identity_sha
                    or saved_protocol["identity_sha256"] != identity_sha):
                raise ValueError("Existing experiment uses different settings or files. Choose a new output directory.")
        else:
            if list(output.glob("run_*.json")) or list((output / "checkpoints").glob("*.json")):
                raise ValueError("Existing outputs have no matching experiment protocol.")
            helper.atomic_json(protocol_path, {"identity_sha256": identity_sha, "identity": identity})
        records = restore_questions(output, identity_sha, queries)
        write_summary(output, args.retriever, queries, records)
        attempts = output / "attempts.jsonl"
        with attempts.open("a") as stream:
            stream.write(json.dumps({"started": datetime.now(timezone.utc).isoformat(),
                                     "slurm_job_id": os.getenv("SLURM_JOB_ID"), "server": server,
                                     "gpu": torch.cuda.get_device_name(0), "saved_questions": len(records)}) + "\n")
        if len(records) == len(queries):
            if not any(row["searches"] for row in records.values()):
                raise RuntimeError("The saved outcomes contain no retrieval activity; inspect this experiment.")
            print("All question outcomes are already saved; no agent work repeated.", flush=True)
            return
        print(f"Loading {args.retriever}; {len(records)}/{len(queries)} outcomes already saved.", flush=True)
        searcher = helper.saved_searcher_class(original_searcher, project, cache, index_folder)(settings)
        if not (project / "Qwen3/Qwen3-Embedding-0.6B/tokenizer_config.json").is_file():
            raise FileNotFoundError("The existing snippet tokenizer is missing.")
        search_tool = tools.SearchToolHandler(searcher, snippet_max_tokens=64, k=10)
        document_tool = tools.GetDocumentToolHandler(searcher)
        Agent = helper.paper_agent_class(original_agent, OpenAI)

        def make_agent():
            agent = Agent(llm={"model": str(agent_model), "model_type": "qwen_dashscope",
                              "generate_cfg": {"temperature": 0.85, "top_p": 0.95,
                                               "presence_penalty": 1.1, "max_input_tokens": 100000}},
                          function_list=["search", "get_document"],
                          search_tool_handler=search_tool, get_document_handler=document_tool)
            agent.context_limit = server["max_model_len"]
            agent.request_budgets, agent.last_request_messages = [], []
            return agent

        summary = execute_questions(output, identity_sha, queries, records, args.retriever, settings,
                                    make_agent, searcher, helper.original_serializer(repo), args.port, server["id"])
        if not any(row["searches"] for row in records.values()):
            raise RuntimeError("No retrieval activity was recorded; inspect these outcomes.")
        print(f"SUCCESS: saved {summary['processed']}/{len(queries)} question outcomes; "
              f"{summary['answer_text_present']} have final-answer text. Correctness awaits judging.", flush=True)
        print("Summary:", output / "sample-summary.json", flush=True)


if __name__ == "__main__":
    main()
