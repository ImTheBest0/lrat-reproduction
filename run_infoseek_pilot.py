#!/usr/bin/env python3
"""Run five official questions with LRAT's agent loop and saved exact indexes."""
import argparse
import ast
import csv
import fcntl
import importlib.util
import json
import logging
import os
import pickle
import re
import sys
import time
import types
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from encode_wiki import DEFAULT_PROJECT, atomic_json, corpus_info, sha256_file, source_hashes
from encode_wiki_optimized import RUN_NAME

PROTOCOL = {
    "temperature": 0.85, "top_p": 0.95, "presence_penalty": 1.1,
    "seed": 2025, "k": 10, "snippet_max_tokens": 64,
    "document_max_tokens": 512, "max_main_loop_calls": 50,
    "query_max_length": 8192, "pooling": "eos", "normalize": True,
    "query_precision": "float16", "query_attention": "flash_attention_2",
    "query_use_cache": False, "request_retry_limit": 3,
}
SOURCE_HASHES = {
    "search_agent/webexplorer_utils/react_agent.py": "3dc494b8ed64546b61a744249700450baccac440e9903b82114bcdf4221a4256",
    "search_agent/webexplorer_utils/tool_search.py": "faa182807757a4445dc5b4b10c10933e34855a3bd40af87c331fe8dea320149f",
    "search_agent/agentcmp_client.py": "bb38efd36fdb18a2ce0d0b8111c354a32673b91e71b42bbcc46e26de359117be",
    "searcher/searchers/faiss_searcher.py": "c140d55439b4ec4d0a5d5274400bee2ca5eb8842b0abece0586d4f60c270e80c",
    "searcher/searchers/base.py": "694e00a8909efe56f48896c567a47c8913219adff1ad05d0fb49b41fa683af6c",
}


def read_queries(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))
    if any(len(row) != 2 or not all(field.strip() for field in row) for row in rows):
        raise ValueError("Queries must contain exactly two nonempty fields: ID and question.")
    rows = [(row[0].strip(), row[1].strip()) for row in rows]
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError("Repeated query IDs.")
    return rows


def assess_answer(prediction, termination, error=None, retrieval_errors=None, finish_reason=None):
    """Classify saved output, without changing generation or judging correctness."""
    text = re.sub(r"<think>.*?</think>", "", prediction or "", flags=re.DOTALL).strip()
    if error:
        status = "request_error"
    elif retrieval_errors:
        status = "retrieval_error"
    elif "<think>" in text:
        status = "unfinished_thinking"
    else:
        # Some chat templates prefill <think> in the prompt, so a generated
        # closing tag can appear without its opening tag in the response.
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[1].strip()
        if "<tool_call>" in text or "</tool_call>" in text:
            status = "tool_call_instead_of_answer"
        elif "<answer>" in text and "</answer>" not in text:
            status = "unfinished_answer"
        else:
            match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
            if match:
                text = match.group(1).strip()
            if not text:
                status = "empty_answer"
            elif finish_reason == "length":
                status = "generation_length_limit"
            elif termination not in ("answer", "generate an answer as token limit reached"):
                status = "unexpected_termination"
            else:
                status = "answer_text_present"
    completed = status == "answer_text_present"
    return {"final_answer": text if completed else "", "answer_status": status,
            "completed_with_answer": completed}


def audit_summary(path):
    """Print a compact audit of an existing summary; never rewrite run files."""
    data = json.loads(path.read_text())
    records = []
    for row in data["results"]:
        budgets = row.get("request_budgets") or []
        finish_reason = budgets[-1].get("finish_reason") if budgets else None
        assessment = assess_answer(
            row.get("final_answer", ""), row.get("termination"), row.get("error"),
            row.get("retrieval_errors"), finish_reason,
        )
        if "answer_status" in row:
            # Newer summaries already classified the raw response, which is
            # retained in the original trajectory rather than repeated here.
            assessment["answer_status"] = row["answer_status"]
            assessment["completed_with_answer"] = row["completed_with_answer"]
        records.append({
            "query_id": row["query_id"], "answer_status": assessment["answer_status"],
            "previous_completed_flag": row.get("completed_with_answer"),
            "completed_with_answer": assessment["completed_with_answer"],
            "tool_calls": row.get("tool_calls", {}), "termination": row.get("termination"),
            "minutes": round(row["seconds"] / 60, 2),
            "finish_reason": finish_reason,
        })
    return {
        "purpose": "Output-format audit only; answer correctness is not judged.",
        "retriever": data["retriever"], "processed": len(records),
        "answer_text_present": sum(row["completed_with_answer"] for row in records),
        "minutes": round(sum(row["seconds"] for row in data["results"]) / 60, 2),
        "note": "Older summaries did not record finish_reason; null means unknown.",
        "results": records,
    }


class ArrowDocuments:
    """Keep text in the memory-mapped Arrow cache; map IDs to dataset rows."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.positions = {}
        ids = dataset.select_columns(["docid"])
        for start in range(0, len(dataset), 100000):
            batch = ids[start:start + 100000]["docid"]
            self.positions.update((str(docid), start + offset) for offset, docid in enumerate(batch))
        if len(self.positions) != len(dataset):
            raise ValueError("The corpus has duplicate document IDs.")

    def __len__(self):
        return len(self.positions)

    def get(self, docid, default=None):
        position = self.positions.get(str(docid))
        if position is None:
            return default
        row = self.dataset[position]
        if str(row["docid"]) != str(docid):
            raise ValueError("Document-ID lookup lost alignment with the corpus.")
        return row["text"]


def saved_searcher_class(original, project, cache, folder):
    """Replace only asset loading; keep the original query/search/document methods."""
    class SavedSearcher(original):
        def _load_model(self):
            import torch
            from transformers import AutoTokenizer
            from tevatron.retriever.modeling import DenseModel
            # The original loader checks 'e5' anywhere in the filesystem path.
            # The pinned Qwen revision hash contains that text, so it accidentally
            # chooses eager attention. Select the tested backend explicitly.
            print("Loading pinned query model with FlashAttention 2:", self.args.model_name, flush=True)
            self.model = DenseModel.load(
                self.args.model_name, pooling=self.args.pooling,
                normalize=self.args.normalize, lora_name_or_path=None,
                cache_dir=os.getenv("HF_HOME"), local_files_only=True,
                torch_dtype=torch.float16, attn_implementation="flash_attention_2",
            )
            self.model.encoder.config.use_cache = False
            self.model = self.model.to("cuda").eval()
            attention = self.model.encoder.config._attn_implementation
            if attention != "flash_attention_2":
                raise RuntimeError(f"Expected FlashAttention 2; model selected {attention!r}.")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_name, cache_dir=os.getenv("HF_HOME"),
                padding_side="left", local_files_only=True,
            )
            print("Query model ready; attention:", attention, "; model cache disabled.", flush=True)

        def _load_dataset(self):
            from tevatron.retriever.arguments import DataArguments
            from tevatron.retriever.dataset import EncodeDataset
            dataset = EncodeDataset(DataArguments(
                dataset_name="json", dataset_path=cache["corpus_path"],
                dataset_cache_dir=str(project / "hf-cache/datasets"), num_proc=2,
            )).encode_data
            if len(dataset) != cache["documents"] or dataset._fingerprint != cache["dataset_fingerprint"]:
                raise ValueError("The opened corpus does not match the encoding cache.")
            print("Preparing document-ID lookup from the existing Arrow cache.", flush=True)
            self.docid_to_text = ArrowDocuments(dataset)

        def _load_faiss_index(self):
            import faiss
            import numpy as np
            from tevatron.retriever.searcher import FaissFlatSearcher
            saved = json.loads((folder / "metadata.json").read_text())
            run = json.loads((project / "encodings" / RUN_NAME / self.args.retriever / "run.json").read_text())
            if saved["build_identity"]["encoding_run"] != run:
                raise ValueError("The saved index belongs to a different encoding run.")
            if (run["model"] != cache["models"][self.args.retriever]
                    or run["corpus_sha256"] != cache["corpus_sha256"]
                    or saved["documents"] != cache["documents"] or saved["dimension"] != 1024
                    or saved["build_identity"]["faiss_version"] != faiss.__version__):
                raise ValueError("Wrong model, corpus, or dimension for this saved index.")
            for filename in ("index.faiss", "docids.pkl"):
                record = saved["files"][filename]
                path = folder / filename
                print("Verifying saved file:", path, flush=True)
                if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
                    raise ValueError(f"Incomplete or damaged index file: {path}")
            self.retriever = FaissFlatSearcher(np.empty((0, 1024), dtype=np.float32))
            self.retriever.index = faiss.read_index(str(folder / "index.faiss"))
            with (folder / "docids.pkl").open("rb") as stream:
                self.lookup = pickle.load(stream)
            index = self.retriever.index
            if (not isinstance(index, faiss.IndexFlatIP) or index.d != 1024
                    or index.ntotal != cache["documents"] or len(self.lookup) != index.ntotal
                    or index.metric_type != faiss.METRIC_INNER_PRODUCT
                    or any(str(docid) not in self.docid_to_text.positions for docid in self.lookup)):
                raise ValueError("The FAISS index and document lookup do not match the corpus.")
            # The working package is faiss-cpu, so vectors stay in host RAM.
            self._setup_gpu()
            self.search_log, self.document_log, self.retrieval_errors = [], [], []

        def search(self, query, k=10):
            started = time.perf_counter()
            try:
                hits = super().search(query, k)
            except Exception as error:
                self.retrieval_errors.append(str(error))
                raise
            self.search_log.append({
                "query": query, "seconds": time.perf_counter() - started,
                "hits": [{"docid": str(hit["docid"]), "score": hit["score"]} for hit in hits],
            })
            print("Search:", query, "| top document:", hits[0]["docid"] if hits else None, flush=True)
            return hits

        def get_document(self, docid):
            try:
                result = super().get_document(docid)
            except Exception as error:
                self.retrieval_errors.append(str(error))
                raise
            self.document_log.append({"docid": str(docid), "found": result is not None})
            print("Read document:", docid, "| found:", result is not None, flush=True)
            return result
    return SavedSearcher


def paper_agent_class(original, client_factory):
    """Keep the original agent loop; send the paper's actual sampling parameters."""
    class PaperAgent(original):
        def call_server(self, msgs, planning_port, max_tries=3):
            self.last_request_messages = [dict(message) for message in msgs]
            inputs = len(self.tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            ))
            requested = int(self.max_generation)
            if inputs + requested + 256 > self.context_limit:
                raise RuntimeError("Request exceeds the server context; completion allowance was not shortened.")
            budget = {"input_tokens": inputs, "completion_allowance": requested}
            self.request_budgets.append(budget)
            error = None
            with client_factory(api_key="EMPTY", base_url=f"http://127.0.0.1:{planning_port}/v1",
                                timeout=600.0, max_retries=0) as client:
                for attempt in range(min(max_tries, PROTOCOL["request_retry_limit"])):
                    try:
                        response = client.chat.completions.create(
                            model=self.model, messages=msgs,
                            stop=["\n<tool_response>", "<tool_response>"],
                            temperature=PROTOCOL["temperature"], top_p=PROTOCOL["top_p"],
                            presence_penalty=PROTOCOL["presence_penalty"],
                            seed=PROTOCOL["seed"], logprobs=True, max_tokens=requested,
                            extra_body={"chat_template_kwargs": {"enable_thinking": self.enable_thinking}},
                        )
                        choice = response.choices[0]
                        budget["finish_reason"] = choice.finish_reason
                        if choice.message.content and choice.message.content.strip():
                            return choice.message.content.strip(), choice.finish_reason
                        error = RuntimeError("Empty model response.")
                    except Exception as caught:
                        if getattr(caught, "status_code", None) in (400, 401, 403, 404, 422):
                            raise
                        error = caught
                    if attempt + 1 < min(max_tries, PROTOCOL["request_retry_limit"]):
                        time.sleep(2 ** attempt)
            raise RuntimeError(f"Model request failed after retries: {error}") from error
    return PaperAgent


def original_serializer(repo):
    path = repo / "search_agent/agentcmp_client.py"
    names = {"parse_messages_to_result_array", "persist_response"}
    functions = [node for node in ast.parse(path.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise RuntimeError("The original trajectory serializer changed.")
    for function in functions:
        if function.name == "persist_response":
            function.body.append(ast.Return(value=ast.Name(id="filename", ctx=ast.Load())))
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    namespace = {"json": json, "re": re, "Path": Path, "datetime": datetime,
                 "timezone": timezone, "logger": logging.getLogger("lrat-pilot")}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["persist_response"]


def load_original_modules(repo):
    for relative, expected in SOURCE_HASHES.items():
        if sha256_file(repo / relative) != expected:
            raise ValueError(f"The pinned original source changed: {relative}")
    for relative in ("tevatron/src", "search_agent", ""):
        sys.path.insert(0, str(repo / relative))
    os.environ["MAX_LLM_CALL_PER_RUN"] = "50"
    # Load the FAISS backend directly, avoiding unused Java/BM25 dependencies.
    name = "_lrat_pilot_searchers"
    package = types.ModuleType(name)
    package.__path__ = [str(repo / "searcher/searchers")]
    sys.modules[name] = package
    spec = importlib.util.spec_from_file_location(name + ".faiss_searcher",
                                                  repo / "searcher/searchers/faiss_searcher.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from webexplorer_utils import react_agent, tool_search
    if react_agent.MAX_LLM_CALL_PER_RUN != 50:
        raise RuntimeError("Expected the original 50-call evaluation loop.")
    return module.FaissSearcher, react_agent.MultiTurnReactAgent, tool_search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--audit-summary", type=Path, nargs="+",
                        help="Print compact saved-summary audits without loading models or using a GPU.")
    parser.add_argument("--retriever", choices=("qwen", "lrat"))
    parser.add_argument("--port", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.audit_summary:
        for path in args.audit_summary:
            print(json.dumps(audit_summary(path), indent=2))
        return
    if args.retriever is None or args.port is None or args.output_dir is None:
        parser.error("A GPU pilot requires --retriever, --port, and --output-dir.")
    project = args.project.resolve()
    os.chdir(project)
    repo = project.parent / "LRAT"
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    import faiss
    import torch
    import flash_attn
    import transformers
    import datasets
    import pyarrow
    from openai import OpenAI
    if (torch.cuda.device_count() != 1 or flash_attn.__version__ != "2.8.3"
            or torch.__version__.split("+")[0] != "2.7.0"
            or transformers.__version__ != "4.53.2"):
        raise RuntimeError("Use one allocated GPU and the tested FlashAttention environment.")
    if faiss.get_num_gpus() != 0:
        raise RuntimeError("This pilot expects the existing faiss-cpu package.")
    faiss.omp_set_num_threads(int(os.getenv("OMP_NUM_THREADS", "8")))
    queries_path = project / "data/infoseek/pilot-5.tsv"
    queries = read_queries(queries_path)
    if len(queries) != 5 or queries != read_queries(project / "data/infoseek/queries.tsv")[:5]:
        raise ValueError("The pilot must be the first five official benchmark questions.")
    _, metadata = corpus_info(project)
    cache = json.loads((project / "data/wiki-25/encoding-cache.json").read_text())
    if (cache["documents"] != metadata["documents"]
            or cache["corpus_sha256"] != metadata["corpus_sha256"]
            or cache["encoder_source_hashes"] != source_hashes(repo)
            or cache["datasets_version"] != datasets.__version__
            or cache["pyarrow_version"] != pyarrow.__version__):
        raise ValueError("The prepared corpus or original encoder changed.")
    for record in cache["cache_files"]:
        if Path(record["filename"]).stat().st_size != record["bytes"]:
            raise ValueError("An Arrow cache file is missing or incomplete.")
    model = cache["models"][args.retriever]
    model_path = Path(model["path"])
    stat = (model_path / "model.safetensors").stat()
    if (sha256_file(model_path / "config.json") != model["config_sha256"]
            or stat.st_size != model["weights_bytes"] or stat.st_mtime_ns != model["weights_mtime_ns"]):
        raise ValueError("The pinned retriever files changed.")
    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=10) as response:
        server = json.load(response)["data"][0]
    context_limit = server.get("max_model_len")
    agent_model = project / "models/AgentCPM-Explore"
    if server["id"] != str(agent_model) or not isinstance(context_limit, int) or context_limit < 120000:
        raise ValueError("Expected AgentCPM with the pilot's larger server context.")
    original_searcher, original_agent, tools = load_original_modules(repo)
    serializer = original_serializer(repo)
    index_folder = project / "indexes" / RUN_NAME / args.retriever
    settings = SimpleNamespace(
        retriever=args.retriever, index_path=str(index_folder),
        model_name=str(model_path), normalize=True, pooling="eos", torch_dtype="float16",
        dataset_name=cache["corpus_path"],
        task_prefix="Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
        max_length=8192, model=str(agent_model), temperature=0.85, top_p=0.95,
        presence_penalty=1.1, snippet_max_tokens=64, k=10, searcher_type="faiss",
    )
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (output / "protocol.json").exists() or list(output.glob("run_*.json")):
            raise ValueError("Use a new output directory for this pilot; existing results will not be overwritten.")
        atomic_json(output / "protocol.json", {
            "purpose": "Five-question operational pilot; correctness has not been judged.",
            "protocol": PROTOCOL, "server": server,
            "queries_sha256": sha256_file(queries_path), "queries": queries,
            "agent_config_sha256": sha256_file(agent_model / "config.json"),
            "agent_generation_config": (
                json.loads((agent_model / "generation_config.json").read_text())
                if (agent_model / "generation_config.json").exists() else None
            ),
            "retriever": model, "index_folder": str(index_folder),
            "corpus_sha256": cache["corpus_sha256"], "source_hashes": SOURCE_HASHES,
            "runner_sha256": sha256_file(Path(__file__)), "slurm_job_id": os.getenv("SLURM_JOB_ID"),
            "runtime_versions": {
                "torch": torch.__version__, "transformers": transformers.__version__,
                "faiss": faiss.__version__, "flash_attention": flash_attn.__version__,
                "datasets": datasets.__version__, "pyarrow": pyarrow.__version__,
            },
        })
        print(f"Loading {args.retriever}: {cache['documents']:,} indexed passages.", flush=True)
        searcher = saved_searcher_class(original_searcher, project, cache, index_folder)(settings)
        # The original tools use this relative tokenizer directory, prepared by the earlier smoke test.
        if not (project / "Qwen3/Qwen3-Embedding-0.6B/tokenizer_config.json").is_file():
            raise FileNotFoundError("The local snippet tokenizer from the smoke test is missing.")
        search_tool = tools.SearchToolHandler(searcher, snippet_max_tokens=64, k=10)
        document_tool = tools.GetDocumentToolHandler(searcher)
        Agent = paper_agent_class(original_agent, OpenAI)
        records = []
        for number, (qid, question) in enumerate(queries, start=1):
            print(f"Question {number}/5: {qid} | {question}", flush=True)
            searcher.search_log.clear(); searcher.document_log.clear(); searcher.retrieval_errors.clear()
            agent = Agent(llm={"model": str(agent_model), "model_type": "qwen_dashscope",
                              "generate_cfg": {"temperature": 0.85, "top_p": 0.95,
                                               "presence_penalty": 1.1, "max_input_tokens": 100000}},
                          function_list=["search", "get_document"],
                          search_tool_handler=search_tool, get_document_handler=document_tool)
            agent.context_limit = context_limit
            agent.request_budgets, agent.last_request_messages = [], []
            started = time.perf_counter()
            try:
                result = agent._run({"item": {"question": question, "answer": ""},
                                     "planning_port": args.port}, server["id"])
            except Exception as error:
                result = {"prediction": "", "termination": "request_error", "error": str(error),
                          "messages": agent.last_request_messages, "tool_call_counts": {},
                          "tool_call_counts_all": {}, "retrieved_docids": []}
            trajectory = serializer(output, qid, question, result, settings)
            finish_reason = agent.request_budgets[-1].get("finish_reason") if agent.request_budgets else None
            assessment = assess_answer(result.get("prediction", ""), result.get("termination"),
                                       result.get("error"), searcher.retrieval_errors, finish_reason)
            final, completed = assessment["final_answer"], assessment["completed_with_answer"]
            record = {
                "query_id": qid, "question": question, "final_answer": final,
                "answer_status": assessment["answer_status"],
                "completed_with_answer": completed, "termination": result.get("termination"),
                "error": result.get("error"), "retrieval_errors": list(searcher.retrieval_errors),
                "tool_calls": result.get("tool_call_counts", {}),
                "searches": list(searcher.search_log), "documents_read": list(searcher.document_log),
                "request_budgets": agent.request_budgets, "seconds": time.perf_counter() - started,
                "trajectory": trajectory.name,
            }
            records.append(record)
            atomic_json(output / "pilot-summary.json", {
                "purpose": "Operational pilot; no accuracy score or LLM judging.",
                "retriever": args.retriever, "questions": 5, "processed": len(records),
                "completed_with_answer": sum(row["completed_with_answer"] for row in records),
                "results": records, "protocol": PROTOCOL, "server_context_limit": context_limit,
            })
            print("Final answer:", final, flush=True)
            print("Answer status:", record["answer_status"], flush=True)
            print("Termination:", record["termination"], "| tool calls:", record["tool_calls"], flush=True)
        completed = sum(row["completed_with_answer"] for row in records)
        print(f"PILOT {args.retriever}: {completed}/5 runs completed with an answer.", flush=True)
        print("Summary:", output / "pilot-summary.json", flush=True)
        if completed != 5 or not any(row["searches"] for row in records):
            raise SystemExit("Pilot needs inspection: failed runs or no retrieval activity.")
        print(f"SUCCESS: {args.retriever} full-corpus pilot ran; answers await judging.", flush=True)


if __name__ == "__main__":
    main()
