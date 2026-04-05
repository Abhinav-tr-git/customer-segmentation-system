import argparse
import sys

from src.pipelines.training_pipeline import run_training
from src.pipelines.inference_pipeline import run_inference


def main():
    parser = argparse.ArgumentParser(
        description="Customer Segmentation Production Pipeline",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "infer"],
        required=True,
        help="Mode to run the pipeline",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input CSV path (optional for train; required for infer)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file",
    )

    args = parser.parse_args()

    if args.mode == "train":
        run_training(config_path=args.config, input_path=args.input)

    elif args.mode == "infer":
        if not args.input:
            print("Error: --input is required for inference mode.")
            sys.exit(1)
        results = run_inference(args.input, args.config)
        print("Inference Results (Sample):")
        print(results.head())


if __name__ == "__main__":
    main()
