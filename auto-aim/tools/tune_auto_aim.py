#!/usr/bin/env python3

import argparse
import copy
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRIC_FIELDS = [
    "frames",
    "detected_frames",
    "tracked_frames",
    "controlled_frames",
    "detection_rate",
    "tracking_rate",
    "control_rate",
    "tracker_loss_events",
    "processing_mean_ms",
    "reprojection_samples",
    "reprojection_rmse_mean_px",
    "reprojection_rmse_std_px",
    "yaw_correction_mean_deg",
    "yaw_correction_std_deg",
    "armor_yaw_step_rms_deg",
    "command_yaw_step_rms_deg",
    "nis_updates",
    "nis_mean",
    "nis_rejection_rate",
]


def resolve_path(path):
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (Path.cwd() / value).resolve()


def parse_parameter(specification):
    if "=" not in specification:
        raise ValueError(f"参数必须使用 key=value1,value2 格式: {specification}")
    key, raw_values = specification.split("=", 1)
    key = key.strip()
    values = [yaml.safe_load(value.strip()) for value in raw_values.split(",")]
    if not key or not values or any(value is None for value in values):
        raise ValueError(f"参数名和值不能为空: {specification}")
    return key, values


def set_dotted_key(config, dotted_key, value):
    keys = dotted_key.split(".")
    current = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            raise KeyError(f"基础配置中不存在参数路径: {dotted_key}")
        current = current[key]
    if keys[-1] not in current:
        raise KeyError(f"基础配置中不存在参数: {dotted_key}")
    current[keys[-1]] = value


def calculate_score(metrics):
    return (
        100.0 * (1.0 - metrics["detection_rate"])
        + 100.0 * (1.0 - metrics["tracking_rate"])
        + metrics["command_yaw_step_rms_deg"]
        + 0.5 * metrics["armor_yaw_step_rms_deg"]
        + 0.2 * metrics["reprojection_rmse_mean_px"]
        + 10.0 * metrics["nis_rejection_rate"]
        + 0.02 * metrics["processing_mean_ms"]
    )


def write_csv(path, results, parameter_names):
    fields = ["rank", "run", "status", "score", "returncode", "dataset"]
    fields += parameter_names + METRIC_FIELDS + ["config", "metrics", "log"]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {
                "rank": result.get("rank", ""),
                "run": result["run"],
                "status": result["status"],
                "score": result.get("score", ""),
                "returncode": result.get("returncode", ""),
                "dataset": result["dataset"],
                "config": result["config"],
                "metrics": result["metrics_path"],
                "log": result["log"],
            }
            row.update(result["parameters"])
            metrics = result.get("metrics", {})
            row.update({name: metrics[name] for name in METRIC_FIELDS if name in metrics})
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="批量回放并评分 auto-aim 参数组合")
    parser.add_argument("--binary", default="build/auto_aim_test")
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output", default="tuning_results")
    parser.add_argument("--param", action="append", required=True)
    parser.add_argument("--end-index", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-runs", type=int, default=128)
    args = parser.parse_args()

    try:
        parameters = [parse_parameter(specification) for specification in args.param]
    except ValueError as error:
        parser.error(str(error))

    parameter_names = [name for name, _ in parameters]
    if len(parameter_names) != len(set(parameter_names)):
        parser.error("同一个参数不能重复指定")

    combinations = list(itertools.product(*(values for _, values in parameters)))
    run_count = len(combinations) * len(args.dataset)
    if run_count > args.max_runs:
        parser.error(
            f"本次将运行 {run_count} 组，超过 --max-runs={args.max_runs}；"
            "请缩小参数范围或显式提高限制"
        )

    binary = resolve_path(args.binary)
    base_config_path = resolve_path(args.base_config)
    datasets = [resolve_path(dataset) for dataset in args.dataset]
    output_dir = resolve_path(args.output)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    if not binary.is_file():
        parser.error(f"找不到回放程序: {binary}")
    if not base_config_path.is_file():
        parser.error(f"找不到基础配置: {base_config_path}")
    for dataset in datasets:
        if not Path(f"{dataset}.avi").is_file() or not Path(f"{dataset}.txt").is_file():
            parser.error(f"找不到成对的录像文件: {dataset}.avi/.txt")

    with base_config_path.open(encoding="utf-8") as source:
        base_config = yaml.safe_load(source)

    results = []
    run_index = 0
    for combination in combinations:
        parameter_values = dict(zip(parameter_names, combination))
        for dataset in datasets:
            run_index += 1
            run_dir = runs_dir / f"{run_index:04d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            config_path = run_dir / "config.yaml"
            metrics_path = run_dir / "metrics.json"
            log_path = run_dir / "run.log"

            config = copy.deepcopy(base_config)
            try:
                for name, value in parameter_values.items():
                    set_dotted_key(config, name, value)
            except KeyError as error:
                parser.error(str(error))
            with config_path.open("w", encoding="utf-8") as output:
                yaml.safe_dump(config, output, allow_unicode=True, sort_keys=False)

            command = [
                str(binary),
                "--display=false",
                f"--config-path={config_path}",
                f"--metrics-out={metrics_path}",
            ]
            if args.end_index > 0:
                command.append(f"--end-index={args.end_index}")
            command.append(str(dataset))

            result = {
                "run": run_index,
                "status": "failed",
                "parameters": parameter_values,
                "dataset": str(dataset),
                "config": str(config_path),
                "metrics_path": str(metrics_path),
                "log": str(log_path),
                "command": command,
            }
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    process = subprocess.run(
                        command,
                        cwd=PROJECT_ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=args.timeout,
                        check=False,
                    )
                result["returncode"] = process.returncode
                if process.returncode == 0 and metrics_path.is_file():
                    with metrics_path.open(encoding="utf-8") as metrics_file:
                        metrics = json.load(metrics_file)
                    result["metrics"] = metrics
                    if metrics["reprojection_samples"] == 0:
                        result["status"] = "no_measurements"
                    else:
                        result["score"] = calculate_score(metrics)
                        result["status"] = "ok"
            except subprocess.TimeoutExpired:
                result["status"] = "timeout"

            results.append(result)
            score_text = f"{result['score']:.4f}" if "score" in result else "-"
            print(f"[{run_index}/{run_count}] {result['status']}: score={score_text}")

    results.sort(key=lambda item: (item["status"] != "ok", item.get("score", float("inf"))))
    for rank, result in enumerate((item for item in results if item["status"] == "ok"), start=1):
        result["rank"] = rank

    summary = {
        "schema_version": 1,
        "base_config": str(base_config_path),
        "score_direction": "lower_is_better",
        "score_formula": (
            "100*(1-detection_rate) + 100*(1-tracking_rate) + "
            "command_yaw_step_rms_deg + 0.5*armor_yaw_step_rms_deg + "
            "0.2*reprojection_rmse_mean_px + 10*nis_rejection_rate + "
            "0.02*processing_mean_ms"
        ),
        "results": results,
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write("\n")
    write_csv(output_dir / "results.csv", results, parameter_names)

    succeeded = sum(result["status"] == "ok" for result in results)
    print(f"完成 {succeeded}/{run_count} 组，结果目录: {output_dir}")
    return 0 if succeeded == run_count else 1


if __name__ == "__main__":
    sys.exit(main())
