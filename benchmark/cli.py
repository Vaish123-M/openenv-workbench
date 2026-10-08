from __future__ import annotations

import argparse
import json
from pathlib import Path

from .models import BenchmarkConfig
from .runner import BenchmarkRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled OpenEnv benchmarks")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--environment", required=True)
    run_parser.add_argument("--tasks", nargs="+", required=True)
    run_parser.add_argument("--agent", default="mock")
    run_parser.add_argument("--runs", type=int, default=1)
    run_parser.add_argument("--max-steps", type=int, default=10)
    run_parser.add_argument("--timeout", type=float)
    run_parser.add_argument("--seed", type=int)
    run_parser.add_argument("--actions", type=Path, help="JSON mapping of task IDs to mock actions")
    run_parser.add_argument("--output-json", type=Path)
    run_parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    if args.command == "run":
        actions = json.loads(args.actions.read_text(encoding="utf-8")) if args.actions else {}
        benchmark = BenchmarkRunner().run(
            BenchmarkConfig(
                environment=args.environment,
                task_ids=args.tasks,
                agent=args.agent,
                runs=args.runs,
                max_steps=args.max_steps,
                timeout=args.timeout,
                seed=args.seed,
                actions=actions,
            )
        )
        if args.output_json:
            BenchmarkRunner.export_json(benchmark, args.output_json)
        if args.output_csv:
            BenchmarkRunner.export_csv(benchmark, args.output_csv)
        print(benchmark.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
