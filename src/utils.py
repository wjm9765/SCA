# src/utils.py
import os
import yaml
from swift.llm import TrainArguments

def load_train_args(config_path: str) -> TrainArguments:
    """
    config.yaml 파일을 읽어 trainargs 객체로 변환
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    args = TrainArguments(**config_dict)
    
    return args

def setup_environment():
    """환경 변수 및 디렉토리 설정
    네트워크 불륨을 위한 workspace로 경로 지정
    """
    os.environ["MODELSCOPE_CACHE"] = "/workspace/modelscope_cache"
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    
    os.makedirs("/workspace/tmp", exist_ok=True)
    os.makedirs("/workspace/modelscope_cache", exist_ok=True)