import argparse
import os
import glob
from swift.llm import sft_main
# src 폴더 내 모듈 임포트
from src.utils import setup_environment, load_train_args
from src.template import register_custom_template
from src.dataset import register_custom_dataset


def find_latest_checkpoint(output_root):
    """
    output_root 내의 모든 v* 폴더를 뒤져서 가장 최신(시간 기준) 체크포인트를 찾습니다.
    """
    # 1. output_root가 실제로 존재하는지 확인
    if not os.path.exists(output_root):
        print(f"debug: Root path not found -> {output_root}")
        return None

    # 2. v* 로 시작하는 모든 폴더 찾기 (예: v0-2025..., v1-2025...)
    version_dirs = glob.glob(os.path.join(output_root, "v*"))
    
    if not version_dirs:
        print(f"debug: No version folders found in -> {output_root}")
        return None

    # 3. [핵심] 이름 말고 '수정 시간' 순으로 정렬 (가장 최근에 수정된게 맨 뒤로)
    version_dirs.sort(key=os.path.getmtime)

    # 4. 최신 폴더부터 역순으로 뒤지며 checkpoint 찾기
    for v_dir in reversed(version_dirs):
        checkpoints = glob.glob(os.path.join(v_dir, "checkpoint-*"))
        if checkpoints:
            # 체크포인트는 숫자 기준으로 정렬 (checkpoint-100, checkpoint-200)
            checkpoints.sort(key=lambda x: int(x.split('checkpoint-')[-1]) if x.split('checkpoint-')[-1].isdigit() else -1)
            return checkpoints[-1]  # 가장 큰 숫자 반환

    return None

def main():
    parser = argparse.ArgumentParser(description="SCA Project Training Script")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path or 'true' to auto-detect")
    args = parser.parse_args()

    setup_environment()
    register_custom_template()
    register_custom_dataset("sca_audio_final")

    # 1. Config 로드 (이 시점에서 ms-swift가 output_dir 뒤에 /v1-... 을 붙여버림)
    train_args = load_train_args(args.config)
    
    # 2. [스마트 Resume 로직 수정됨]
    if args.resume:
        if args.resume.lower() == "true":
            search_dir = os.path.dirname(train_args.output_dir)
            
            print(f"🔍 Auto-detecting latest checkpoint in root: {search_dir}...")
            found_ckpt = find_latest_checkpoint(search_dir)
            
            if found_ckpt:
                train_args.resume_from_checkpoint = found_ckpt
                train_args.output_dir = os.path.dirname(found_ckpt)
                train_args.add_version = False
                
                print(f"✅ Found latest checkpoint: {found_ckpt}")
            else:
                print("⚠️  Warning: No existing checkpoints found. Starting from scratch.")
                train_args.resume_from_checkpoint = None
        else:
            # 사용자가 직접 경로 지정한 경우
            train_args.resume_from_checkpoint = args.resume
            if "checkpoint-" in args.resume:
                train_args.output_dir = os.path.dirname(args.resume)

    print("=" * 60)
    print(f"📄 Config: {args.config}")
    
    if train_args.resume_from_checkpoint:
        print(f"🔄 Status: RESUMING TRAINING")
        print(f"   Target: {train_args.resume_from_checkpoint}")
    else:
        print(f"🆕 Status: STARTING NEW TRAINING")
    
    print(f"   OutDir: {train_args.output_dir}")
    print("=" * 60)

    sft_main(train_args)

if __name__ == "__main__":
    main()