# train.py
import os
os.environ["TORCH_FORCE_LOAD_VULNERABLE"] = "1"

import argparse
from swift.llm import sft_main
from src.utils import setup_environment, load_train_args
from src.template import register_custom_template
from src.dataset import register_custom_dataset

def main():
    parser = argparse.ArgumentParser(description="SCA Project Training Script")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    setup_environment()

    register_custom_template()
    register_custom_dataset("sca_audio_final")

    train_args = load_train_args(args.config)
    
    print(f" Loading Config from: {args.config}")
    print(f" Model: {train_args.model}")
    print(f" Output Dir: {train_args.output_dir}")

    # 5. 학습 시작
    sft_main(train_args)

if __name__ == "__main__":
    main()