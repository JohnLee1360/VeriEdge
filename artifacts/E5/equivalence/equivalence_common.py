from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

ROOT = Path(__file__).resolve().parent
INFERENCE_LIB = ROOT.parent / "inference_e2e" / "lib"
if str(INFERENCE_LIB) not in sys.path:
    sys.path.insert(0, str(INFERENCE_LIB))

from common import select_task_prompts, utc_iso_now, write_json  # noqa: E402
from exo_state_utils import first_shard_provider, instance_node_count, iter_model_instances, node_ip_map, unwrap_tagged  # noqa: E402

from score import extract_gsm8k_gold_answer, extract_prediction, is_exact_match, normalize_numeric_answer  # noqa: E402


READY_RUNNER_STATUSES = {"RunnerIdle", "RunnerReady", "RunnerRunning"}
ACTIVE_TASK_STATUSES = {"Pending", "Running"}
SETTING_ORDER = ["1device", "2device", "3device"]
SETTING_LABELS = {
    "1device": "1-device exo",
    "2device": "2-device exo",
    "3device": "3-device exo",
}
SETTING_COLORS = {
    "1device": "#4C78A8",
    "2device": "#72B7B2",
    "3device": "#E6A141",
}


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def output_paths(cfg: Dict[str, Any], setting_name: str) -> Dict[str, Path]:
    output_root = Path(str(cfg["experiment"]["output_root"]))
    run_root = output_root / setting_name
    shared_sample_path = Path(str(cfg["experiment"].get("shared_sample_path", output_root / "sampled_questions.json")))
    return {
        "root": output_root,
        "run_root": run_root,
        "results": run_root / "results.json",
        "sample": shared_sample_path,
        "plot": output_root / "accuracy_comparison.png",
        "check_report": run_root / "check_report.json",
    }


def ensure_output_root(paths: Dict[str, Path]) -> None:
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["run_root"].mkdir(parents=True, exist_ok=True)


def _load_dataset_loader():
    from datasets import load_dataset  # type: ignore

    return load_dataset


def load_or_create_sample(cfg: Dict[str, Any], sample_path: Path) -> List[Dict[str, Any]]:
    if sample_path.exists():
        with sample_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    dataset_cfg = cfg["dataset"]
    load_dataset = _load_dataset_loader()
    dataset = load_dataset(
        path=str(dataset_cfg["hf_dataset"]),
        name=str(dataset_cfg["subset"]),
        split=str(dataset_cfg["split"]),
    )
    prompts = [(idx, row["question"]) for idx, row in enumerate(dataset)]
    selected = select_task_prompts(
        prompts,
        question_count=int(dataset_cfg["sample_size"]),
        seed=int(cfg["experiment"]["seed"]),
    )
    rows: List[Dict[str, Any]] = []
    for dataset_index, question_text in selected:
        example = dataset[int(dataset_index)]
        rows.append(
            {
                "sample_id": len(rows),
                "dataset_index": int(dataset_index),
                "question": str(question_text),
                "gold_answer_text": str(example["answer"]),
                "gold_final_answer": extract_gsm8k_gold_answer(str(example["answer"])),
            }
        )
    write_json(sample_path, rows)
    return rows


def build_prompt(cfg: Dict[str, Any], question: str) -> List[Dict[str, str]]:
    prompt_cfg = cfg["prompt"]
    messages: List[Dict[str, str]] = []
    system_prompt = str(prompt_cfg.get("system_prompt", "")).strip()
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    user_prompt = str(prompt_cfg["user_template"]).format(question=question)
    messages.append({"role": "user", "content": user_prompt})
    return messages


def summarize_state_for_model(state: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    instances_summary: List[Dict[str, Any]] = []
    for instance_id, instance in iter_model_instances(state, model_id):
        first_node_id, first_ip = first_shard_provider(instance, state)
        instances_summary.append(
            {
                "instance_id": instance_id,
                "node_count": instance_node_count(instance),
                "first_shard_node_id": first_node_id,
                "first_shard_ip": first_ip,
                "member_node_ids": sorted(instance_member_node_ids(instance)),
                "member_ips": instance_member_ips(instance, state),
            }
        )
    instances_summary.sort(key=lambda item: item["instance_id"])
    return {
        "captured_at": utc_iso_now(),
        "instance_count": len(instances_summary),
        "instances": instances_summary,
        "cluster_primary_ips": primary_cluster_ips(state),
    }


def primary_cluster_ips(state: Dict[str, Any]) -> List[str]:
    ip_map = node_ip_map(state)
    return sorted({ips[0] for ips in ip_map.values() if ips})


def instance_member_node_ids(instance: Dict[str, Any]) -> List[str]:
    instance = unwrap_tagged(instance)
    shard_assignments = instance.get("shardAssignments")
    if not isinstance(shard_assignments, dict):
        return []
    node_to_runner = shard_assignments.get("nodeToRunner")
    if not isinstance(node_to_runner, dict):
        return []
    return sorted(str(node_id) for node_id in node_to_runner.keys())


def instance_member_ips(instance: Dict[str, Any], state: Dict[str, Any]) -> List[str]:
    ip_map = node_ip_map(state)
    return sorted({ip_map[node_id][0] for node_id in instance_member_node_ids(instance) if node_id in ip_map and ip_map[node_id]})


def state_node_ids_by_primary_ip(state: Dict[str, Any]) -> Dict[str, str]:
    ip_map = node_ip_map(state)
    return {ips[0]: node_id for node_id, ips in ip_map.items() if ips}


def preview_matches_expected_ips(preview: Dict[str, Any], state: Dict[str, Any], expected_ips: Sequence[str]) -> bool:
    instance = preview.get("instance")
    if not isinstance(instance, dict):
        return False
    return instance_member_ips(instance, state) == sorted(str(ip) for ip in expected_ips)


def select_preview_for_expected_ips(
    previews: Sequence[Dict[str, Any]],
    *,
    state: Dict[str, Any],
    model_id: str,
    expected_ips: Sequence[str],
) -> Dict[str, Any]:
    matches = [
        dict(preview)
        for preview in previews
        if preview.get("error") is None
        and str(preview.get("model_id") or preview.get("modelId") or "") == model_id
        and preview_matches_expected_ips(dict(preview), state, expected_ips)
    ]
    if not matches:
        raise RuntimeError(f"no placement preview matched expected_ips={sorted(expected_ips)}")
    if len(matches) == 1:
        return matches[0]
    return sorted(matches, key=_preview_sort_key)[0]


def _preview_sort_key(preview: Dict[str, Any]) -> tuple[str, str, str]:
    command_id = str(preview.get("command_id") or preview.get("commandId") or "")
    instance_id = str(preview.get("instance_id") or preview.get("instanceId") or "")
    preview_blob = json.dumps(preview, sort_keys=True, ensure_ascii=False)
    return command_id, instance_id, preview_blob


def find_matching_instance(
    *,
    state: Dict[str, Any],
    model_id: str,
    expected_ips: Sequence[str],
) -> Optional[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []
    for instance_id, instance in iter_model_instances(state, model_id):
        member_ips = instance_member_ips(instance, state)
        if member_ips == sorted(str(ip) for ip in expected_ips):
            matches.append(
                {
                    "instance_id": instance_id,
                    "instance": instance,
                    "member_ips": member_ips,
                    "member_node_ids": instance_member_node_ids(instance),
                    "node_count": instance_node_count(instance),
                }
            )
    if len(matches) > 1:
        raise RuntimeError(f"multiple live instances matched expected_ips={sorted(expected_ips)}")
    return matches[0] if matches else None


def _unwrap_single_key(value: Any) -> Any:
    while isinstance(value, dict) and len(value) == 1:
        value = next(iter(value.values()))
    return value


def _instance_record(state: Dict[str, Any], instance_id: str) -> Optional[Dict[str, Any]]:
    wrapped = state.get("instances", {}).get(instance_id)
    if not isinstance(wrapped, dict):
        return None
    record = _unwrap_single_key(wrapped)
    return record if isinstance(record, dict) else None


def _task_record(wrapped: Any) -> Optional[Dict[str, Any]]:
    record = _unwrap_single_key(wrapped)
    return record if isinstance(record, dict) else None


def _status_name(value: Any) -> Optional[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and len(value) == 1:
        return next(iter(value.keys()))
    value = _unwrap_single_key(value)
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and len(value) == 1:
        return next(iter(value.keys()))
    return None


def instance_runner_ids(state: Dict[str, Any], instance_id: str) -> List[str]:
    record = _instance_record(state, instance_id)
    if record is None:
        return []
    shard_assignments = record.get("shardAssignments")
    if not isinstance(shard_assignments, dict):
        return []
    node_to_runner = shard_assignments.get("nodeToRunner")
    if not isinstance(node_to_runner, dict):
        return []
    return sorted(str(runner_id) for runner_id in node_to_runner.values())


def active_task_snapshot(state: Dict[str, Any], instance_id: str) -> List[Dict[str, Any]]:
    active: List[Dict[str, Any]] = []
    for task_id, wrapped in state.get("tasks", {}).items():
        task = _task_record(wrapped)
        if task is None:
            continue
        if str(task.get("instanceId") or task.get("instance_id") or "") != instance_id:
            continue
        status = _status_name(task.get("taskStatus") or task.get("task_status"))
        if status not in ACTIVE_TASK_STATUSES:
            continue
        active.append(
            {
                "task_id": str(task_id),
                "task_type": next(iter(wrapped.keys())) if isinstance(wrapped, dict) and len(wrapped) == 1 else None,
                "task_status": status,
                "command_id": str(task.get("commandId") or task.get("command_id") or ""),
            }
        )
    return active


def blocking_task_snapshot(state: Dict[str, Any], instance_id: str) -> List[Dict[str, Any]]:
    return [task for task in active_task_snapshot(state, instance_id) if task.get("task_type") != "DownloadModel"]


def runner_status_snapshot(state: Dict[str, Any], instance_id: str) -> Dict[str, str]:
    runner_statuses: Dict[str, str] = {}
    for runner_id in instance_runner_ids(state, instance_id):
        status = _status_name(state.get("runners", {}).get(runner_id))
        if status is not None:
            runner_statuses[runner_id] = status
    return runner_statuses


def instance_runners_ready(runner_statuses: Dict[str, str]) -> bool:
    if not runner_statuses:
        return False
    return all(status in READY_RUNNER_STATUSES for status in runner_statuses.values())


def task_counts_by_instance(state: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    wrapped_tasks = state.get("tasks", {})
    if not isinstance(wrapped_tasks, dict):
        return counts
    for wrapped in wrapped_tasks.values():
        task = _task_record(wrapped)
        if task is None:
            continue
        instance_id = task.get("instanceId") or task.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            continue
        counts[instance_id] = counts.get(instance_id, 0) + 1
    return counts


def infer_routed_instance(
    *,
    before_state: Dict[str, Any],
    after_state: Dict[str, Any],
    model_id: str,
) -> Dict[str, Any]:
    before_counts = task_counts_by_instance(before_state)
    after_counts = task_counts_by_instance(after_state)
    deltas: List[Dict[str, Any]] = []
    for instance_id, instance in iter_model_instances(after_state, model_id):
        delta = int(after_counts.get(instance_id, 0)) - int(before_counts.get(instance_id, 0))
        deltas.append(
            {
                "instance_id": instance_id,
                "delta": delta,
                "node_count": instance_node_count(instance),
                "first_shard_ip": first_shard_provider(instance, after_state)[1],
                "member_ips": instance_member_ips(instance, after_state),
            }
        )
    deltas.sort(key=lambda item: (item["delta"], item["instance_id"]), reverse=True)
    positives = [item for item in deltas if int(item["delta"]) > 0]
    observed = positives[0] if len(positives) == 1 else None
    return {
        "observed_instance_id": observed["instance_id"] if observed else None,
        "observed_instance_node_count": observed["node_count"] if observed else None,
        "observed_first_shard_ip": observed["first_shard_ip"] if observed else None,
        "observed_member_ips": observed["member_ips"] if observed else None,
        "inference_status": "single_delta_match" if observed else "ambiguous",
        "task_count_deltas": deltas,
    }


def build_initial_results(
    cfg: Dict[str, Any],
    *,
    setting_name: str,
    namespace: str,
    label: str,
    entry_url: str,
    expected_ips: Sequence[str],
    sample_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    questions: List[Dict[str, Any]] = []
    for row in sample_rows:
        questions.append(
            {
                "sample_id": int(row["sample_id"]),
                "dataset_index": int(row["dataset_index"]),
                "question": str(row["question"]),
                "gold_answer_text": str(row["gold_answer_text"]),
                "gold_final_answer": row["gold_final_answer"],
                "result": None,
            }
        )
    return {
        "setting": setting_name,
        "cluster_namespace": namespace,
        "config_snapshot": deepcopy(cfg),
        "run_metadata": {
            "label": label,
            "entry_url": entry_url,
            "expected_member_ips": list(sorted(str(ip) for ip in expected_ips)),
            "started_at": utc_iso_now(),
        },
        "sample_metadata": {
            "seed": int(cfg["experiment"]["seed"]),
            "sample_size": int(cfg["dataset"]["sample_size"]),
            "dataset": str(cfg["dataset"]["hf_dataset"]),
            "subset": str(cfg["dataset"]["subset"]),
            "split": str(cfg["dataset"]["split"]),
            "sampled_at": utc_iso_now(),
        },
        "cluster_evidence": {
            "before_run": None,
            "after_run": None,
            "state_views": [],
        },
        "instance_evidence": {
            "instance_id": None,
            "instance_source": None,
            "preview": None,
            "member_ips": [],
            "member_node_ids": [],
            "ready_snapshot": None,
            "created_command_id": None,
        },
        "summary": {
            "accuracy": None,
            "count": 0,
        },
        "questions": questions,
    }


def load_or_initialize_results(
    cfg: Dict[str, Any],
    *,
    setting_name: str,
    namespace: str,
    label: str,
    entry_url: str,
    expected_ips: Sequence[str],
    sample_rows: Sequence[Dict[str, Any]],
    results_path: Path,
    resume: bool,
) -> Dict[str, Any]:
    if resume and results_path.exists():
        with results_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return build_initial_results(
        cfg,
        setting_name=setting_name,
        namespace=namespace,
        label=label,
        entry_url=entry_url,
        expected_ips=expected_ips,
        sample_rows=sample_rows,
    )


def summarize_accuracy(results: Dict[str, Any]) -> None:
    rows = [q.get("result") for q in results.get("questions", []) if q.get("result")]
    count = len(rows)
    correct = sum(1 for row in rows if row.get("is_correct") is True)
    results["summary"]["count"] = count
    results["summary"]["accuracy"] = (correct / count) if count else None


def persist_results(results: Dict[str, Any], results_path: Path) -> None:
    summarize_accuracy(results)
    write_json(results_path, results)


def record_question_result(
    *,
    question_row: Dict[str, Any],
    label: str,
    entry_url: str,
    namespace: str,
    instance_id: str,
    completion: Dict[str, Any],
    preflight_summary: Dict[str, Any],
    routing_observation: Dict[str, Any],
) -> Dict[str, Any]:
    raw_prediction, extraction_source = extract_prediction(str(completion["output_text"]))
    normalized_prediction = normalize_numeric_answer(raw_prediction)
    gold = question_row.get("gold_final_answer")
    result = {
        "label": label,
        "entry_url": entry_url,
        "cluster_namespace": namespace,
        "instance_id": instance_id,
        "preflight_state_summary": deepcopy(preflight_summary),
        "raw_output": completion["output_text"],
        "predicted_answer": raw_prediction,
        "normalized_answer": normalized_prediction,
        "gold_answer": gold,
        "is_correct": is_exact_match(normalized_prediction, gold),
        "extraction_source": extraction_source,
        "routing_observation": routing_observation,
        "request_metadata": {
            "request_payload": completion["request_payload"],
            "usage": completion["usage"],
            "finish_reason": completion["finish_reason"],
            "completed_at": utc_iso_now(),
        },
    }
    question_row["result"] = result
    return result


def plot_values_from_results(results_by_setting: Dict[str, Dict[str, Any]]) -> List[float]:
    return [float(results_by_setting[name]["summary"].get("accuracy") or 0.0) for name in SETTING_ORDER]
