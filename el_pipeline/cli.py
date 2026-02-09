import argparse
import json
from pathlib import Path

from el_pipeline.config import PipelineConfig
from el_pipeline.pipeline import ELPipeline


def main():
    parser = argparse.ArgumentParser(description="Run modular EL pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to pipeline config JSON file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Input file paths.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional JSONL output path.",
    )
    args = parser.parse_args()

    config_data = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config = PipelineConfig.from_dict(config_data)

    pipeline = ELPipeline(config)
    pipeline.run(args.input, output_path=args.output)


if __name__ == "__main__":
    main()

