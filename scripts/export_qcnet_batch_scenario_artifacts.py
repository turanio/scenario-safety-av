import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from datasets import ArgoverseV2Dataset
from predictors import QCNet
from transforms import TargetBuilder


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_scalar_text(value):
    if isinstance(value, (list, tuple)):
        return str(value[0])
    return str(value)


def flatten_actor_ids(value):
    if torch.is_tensor(value):
        return [str(x.item()) for x in value.flatten()]

    if isinstance(value, np.ndarray):
        return [str(x) for x in value.reshape(-1)]

    if isinstance(value, (list, tuple)):
        if len(value) == 1 and isinstance(value[0], (list, tuple, np.ndarray)):
            return flatten_actor_ids(value[0])

        if all(not isinstance(x, (list, tuple, np.ndarray)) and not torch.is_tensor(x) for x in value):
            return [str(x) for x in value]

        out = []
        for item in value:
            out.extend(flatten_actor_ids(item))
        return out

    return [str(value)]


def read_manifest(path):
    with open(path, encoding="utf-8") as handle:
        scenario_ids = [line.strip() for line in handle if line.strip()]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError(f"Scenario manifest contains duplicate IDs: {path}")
    return scenario_ids


def existing_artifact_row(path, expected_scenario_id):
    with np.load(path, allow_pickle=False) as data:
        scenario_id = get_scalar_text(data["scenario_id"])
        if scenario_id != expected_scenario_id:
            raise RuntimeError(
                f"Existing artifact ID {scenario_id} does not match {expected_scenario_id}"
            )
        probabilities = np.asarray(data["probabilities"], dtype=float).reshape(-1)
        target_actor_id = get_scalar_text(data["target_actor_id"])
        if probabilities.shape != (6,) or not np.isfinite(probabilities).all():
            raise RuntimeError(f"Existing artifact is structurally invalid: {path}")
    return {
        "scenario_id": scenario_id,
        "target_actor_id": target_actor_id,
        "artifact_path": path,
        "top1_mode": int(probabilities.argmax()),
        "top1_probability": float(probabilities.max()),
        "probability_sum": float(probabilities.sum()),
        "status": "resumed_existing",
    }


def export_one(model, data, raw_data, output_dir, device):
    scenario_id = get_scalar_text(data["scenario_id"])

    if isinstance(data, Batch):
        data["agent"]["av_index"] += data["agent"]["ptr"][:-1]

    if isinstance(raw_data, Batch):
        raw_data["agent"]["av_index"] += raw_data["agent"]["ptr"][:-1]

    data = data.to(device)

    with torch.no_grad():
        pred = model(data)

    if model.output_head:
        traj_refine = torch.cat(
            [
                pred["loc_refine_pos"][..., : model.output_dim],
                pred["loc_refine_head"],
                pred["scale_refine_pos"][..., : model.output_dim],
                pred["conc_refine_head"],
            ],
            dim=-1,
        )
    else:
        traj_refine = torch.cat(
            [
                pred["loc_refine_pos"][..., : model.output_dim],
                pred["scale_refine_pos"][..., : model.output_dim],
            ],
            dim=-1,
        )

    pi = pred["pi"]
    eval_mask = data["agent"]["category"] == 3

    if int(eval_mask.sum()) == 0:
        return None

    transformed_actor_ids = flatten_actor_ids(data["agent"]["id"])
    eval_indices = torch.where(eval_mask)[0].cpu().tolist()

    if not eval_indices:
        return None

    focal_idx = int(eval_indices[0])

    if focal_idx >= len(transformed_actor_ids):
        raise RuntimeError(
            f"Focal index {focal_idx} out of range for transformed actor IDs in scenario {scenario_id}"
        )

    target_actor_id = str(transformed_actor_ids[focal_idx])

    origin_eval = data["agent"]["position"][eval_mask, model.num_historical_steps - 1]
    theta_eval = data["agent"]["heading"][eval_mask, model.num_historical_steps - 1]

    cos, sin = theta_eval.cos(), theta_eval.sin()
    rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=theta_eval.device)
    rot_mat[:, 0, 0] = cos
    rot_mat[:, 0, 1] = sin
    rot_mat[:, 1, 0] = -sin
    rot_mat[:, 1, 1] = cos

    traj_eval = (
        torch.matmul(traj_refine[eval_mask, :, :, :2], rot_mat.unsqueeze(1))
        + origin_eval[:, :2].reshape(-1, 1, 1, 2)
    )

    pi_eval = F.softmax(pi[eval_mask], dim=-1)

    positions = traj_eval[0].cpu().numpy().astype(np.float32)
    probabilities = pi_eval[0].cpu().numpy().astype(np.float32)

    raw_positions = raw_data["agent"]["position"][..., :2]
    raw_valid_mask = raw_data["agent"].get("valid_mask", None)

    hist = model.num_historical_steps
    fut = model.num_future_steps

    av_index = raw_data["agent"]["av_index"]
    if torch.is_tensor(av_index):
        av_index = int(av_index.flatten()[0].item())
    else:
        av_index = int(av_index)

    raw_actor_ids = flatten_actor_ids(raw_data["agent"]["id"])

    if target_actor_id not in raw_actor_ids:
        raise RuntimeError(
            f"Target actor {target_actor_id} not found in raw_data for scenario {scenario_id}. "
            f"First raw IDs: {raw_actor_ids[:10]}"
        )

    raw_focal_idx = raw_actor_ids.index(target_actor_id)

    ego_history_positions = to_numpy(raw_positions[av_index, :hist]).astype(np.float32)
    ego_future_positions = to_numpy(raw_positions[av_index, hist:hist + fut]).astype(np.float32)

    target_history_positions = to_numpy(raw_positions[raw_focal_idx, :hist]).astype(np.float32)
    target_future_positions = to_numpy(raw_positions[raw_focal_idx, hist:hist + fut]).astype(np.float32)

    if raw_valid_mask is not None:
        ego_history_valid_mask = to_numpy(raw_valid_mask[av_index, :hist]).astype(bool)
        ego_future_valid_mask = to_numpy(raw_valid_mask[av_index, hist:hist + fut]).astype(bool)
        target_history_valid_mask = to_numpy(raw_valid_mask[raw_focal_idx, :hist]).astype(bool)
        target_future_valid_mask = to_numpy(raw_valid_mask[raw_focal_idx, hist:hist + fut]).astype(bool)
    else:
        ego_history_valid_mask = np.ones(hist, dtype=bool)
        ego_future_valid_mask = np.ones(fut, dtype=bool)
        target_history_valid_mask = np.ones(hist, dtype=bool)
        target_future_valid_mask = np.ones(fut, dtype=bool)

    output_path = os.path.join(output_dir, f"{scenario_id}.npz")

    np.savez(
        output_path,
        scenario_id=np.array(str(scenario_id)),
        target_actor_id=np.array(str(target_actor_id)),
        ego_actor_id=np.array("AV"),
        dt=np.array(0.1, dtype=np.float32),
        positions=positions,
        probabilities=probabilities,
        ego_history_positions=ego_history_positions,
        ego_future_positions=ego_future_positions,
        target_history_positions=target_history_positions,
        target_future_positions=target_future_positions,
        ego_history_valid_mask=ego_history_valid_mask,
        ego_future_valid_mask=ego_future_valid_mask,
        target_history_valid_mask=target_history_valid_mask,
        target_future_valid_mask=target_future_valid_mask,
        coordinate_frame=np.array("av2_global"),
        source=np.array("QCNet_AV2_checkpoint_on_AV2_val_batch_with_ego_and_ground_truth"),
    )

    return {
        "scenario_id": scenario_id,
        "target_actor_id": target_actor_id,
        "artifact_path": output_path,
        "top1_mode": int(probabilities.argmax()),
        "top1_probability": float(probabilities.max()),
        "probability_sum": float(probabilities.sum()),
        "status": "exported",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--scenario_ids_file", required=True)
    parser.add_argument("--expected_num_scenarios", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    started_at = time.time()
    manifest_ids = read_manifest(args.scenario_ids_file)
    if len(manifest_ids) != args.expected_num_scenarios:
        raise RuntimeError(
            f"Manifest has {len(manifest_ids)} IDs; expected {args.expected_num_scenarios}"
        )
    selected_ids = set(manifest_ids)

    existing_npz = [
        name for name in os.listdir(args.output_dir) if name.endswith(".npz")
    ] if os.path.isdir(args.output_dir) else []
    if existing_npz and not args.resume:
        raise RuntimeError(
            f"Output directory already contains {len(existing_npz)} artifacts; "
            "use --resume only after inspecting the prior attempt"
        )
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference was requested, but torch.cuda.is_available() is false")

    model = QCNet.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
        map_location=torch.device("cpu"),
    )
    model.eval()
    model.to(device)

    inference_dataset = ArgoverseV2Dataset(
        root=args.root,
        split="val",
        transform=TargetBuilder(
            num_historical_steps=model.num_historical_steps,
            num_future_steps=model.num_future_steps,
        ),
    )

    raw_dataset = ArgoverseV2Dataset(
        root=args.root,
        split="val",
        transform=None,
    )

    inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False, num_workers=0)
    raw_loader = DataLoader(raw_dataset, batch_size=1, shuffle=False, num_workers=0)

    raw_by_scenario_id = {}
    for raw_data in raw_loader:
        raw_sid = get_scalar_text(raw_data["scenario_id"])
        if raw_sid in selected_ids:
            if raw_sid in raw_by_scenario_id:
                raise RuntimeError(f"Duplicate raw scenario in dataset: {raw_sid}")
            raw_by_scenario_id[raw_sid] = raw_data

    rows = []
    failures = []
    encountered_ids = []
    exported_count = 0
    resumed_count = 0

    for idx, data in enumerate(inference_loader):
        scenario_id = get_scalar_text(data["scenario_id"])
        if scenario_id not in selected_ids:
            continue
        encountered_ids.append(scenario_id)
        raw_data = raw_by_scenario_id.get(scenario_id)

        if raw_data is None:
            print(f"[{idx + 1}] skipped {scenario_id}: raw scenario not found")
            failures.append(
                {"scenario_id": scenario_id, "error": "raw scenario not found"}
            )
            continue

        output_path = os.path.join(args.output_dir, f"{scenario_id}.npz")
        if args.resume and os.path.isfile(output_path):
            try:
                row = existing_artifact_row(output_path, scenario_id)
                rows.append(row)
                resumed_count += 1
                print(f"[{idx + 1}] resumed {scenario_id}")
                continue
            except Exception as exc:
                print(f"[{idx + 1}] existing artifact invalid; retrying {scenario_id}: {exc}")

        try:
            row = export_one(model, data, raw_data, args.output_dir, device)
        except Exception as exc:
            print(f"[{idx + 1}] skipped {scenario_id}: {exc}")
            failures.append({"scenario_id": scenario_id, "error": str(exc)})
            continue

        if row is not None:
            rows.append(row)
            exported_count += 1
            print(
                f"[{idx + 1}] saved {row['scenario_id']} "
                f"target={row['target_actor_id']} "
                f"top1_prob={row['top1_probability']:.4f}"
            )
        else:
            print(f"[{idx + 1}] skipped {scenario_id}: no focal actor")
            failures.append({"scenario_id": scenario_id, "error": "no focal actor"})

    summary_path = os.path.join(args.output_dir, "batch_export_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario_id",
                "target_actor_id",
                "artifact_path",
                "top1_mode",
                "top1_probability",
                "probability_sum",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    failures_path = os.path.join(args.output_dir, "batch_export_failures.csv")
    with open(failures_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario_id", "error"])
        writer.writeheader()
        writer.writerows(failures)

    duplicate_count = len(encountered_ids) - len(set(encountered_ids))
    missing_dataset_ids = sorted(selected_ids - set(encountered_ids))
    audit = {
        "manifest_size": len(manifest_ids),
        "processed_count": len(encountered_ids),
        "success_count": len(rows),
        "new_export_count": exported_count,
        "resumed_existing_count": resumed_count,
        "skip_count": len(failures) + len(missing_dataset_ids),
        "retry_count": exported_count if args.resume else 0,
        "failure_count": len(failures),
        "missing_dataset_count": len(missing_dataset_ids),
        "duplicate_count": duplicate_count,
        "elapsed_seconds": time.time() - started_at,
        "device": str(device),
    }
    audit_path = os.path.join(args.output_dir, "batch_export_audit.json")
    with open(audit_path, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)
        handle.write("\n")

    print(f"Exported {len(rows)} scenario artifacts")
    print(f"Summary saved to: {summary_path}")
    print(f"Audit saved to: {audit_path}")
    for key, value in audit.items():
        print(f"{key}: {value}")

    if missing_dataset_ids:
        print(f"First missing dataset IDs: {missing_dataset_ids[:10]}")
    if len(rows) != len(manifest_ids) or failures or missing_dataset_ids or duplicate_count:
        raise RuntimeError("Export did not produce one valid artifact for every manifest ID")


if __name__ == "__main__":
    main()
