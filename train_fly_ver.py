import os
import sys
import warnings
import numpy as np
import torch
from datasets import Dataset
from swift.llm import sft_main, TrainArguments, register_dataset, DatasetMeta
from sca_data.dataset_utils import easy_load

# -------------------------------------------------------------------------
# [1] 환경 설정
# -------------------------------------------------------------------------
#os.environ["MODELSCOPE_CACHE"] = "/workspace/modelscope_cache"
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#os.environ["TORCH_UES_CUDA_DSA"] = '1'

os.makedirs("/workspace/tmp", exist_ok=True)
os.makedirs("/workspace/modelscope_cache", exist_ok=True)
warnings.filterwarnings("ignore")

#MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_ID = "/workspace/models/huihui_uncensored"
CUSTOM_TEMPLATE = "qwen3-omni-sca"
DATASET_NAME = "sca_audio_final"

def quiet_print(*args):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(*args, flush=True)

# -------------------------------------------------------------------------
# [2] 포맷 변환 함수 (기존 easy_load 출력 -> Qwen 포맷)
# -------------------------------------------------------------------------
def convert_batch_to_qwen_format(batch):
    """
    easy_load가 만든 '표준 Chat 포맷(리스트)'을 
    Qwen이 좋아하는 '<audio> 태그 문자열' 포맷으로 변환합니다.
    (배치 단위 처리)
    """
    ret_messages = []
    ret_audios = []

    # batch["messages"]는 리스트의 리스트입니다. (Batch Size만큼)
    for conversation in batch["messages"]:
        new_conv = []
        new_conv_audios = []

        for msg in conversation:
            role = msg["role"]
            content = msg["content"]
            
            # content가 리스트(멀티모달)인 경우 변환 수행
            if isinstance(content, list):
                new_text = ""
                for item in content:
                    if item["type"] == "text":
                        new_text += item["text"] + "\n"
                    elif item["type"] == "audio":
                        # 1. 오디오 파형 추출
                        # (easy_load가 이미 로드해둔 np.array를 가져옴)
                        wave = item.get("audio_waveform")
                        if isinstance(wave, np.ndarray) and wave.dtype != np.float32:
                            wave = wave.astype(np.float32)
                        new_conv_audios.append(wave)
                        
                        # 2. <audio> 태그 삽입
                        new_text += "<audio>"
                
                new_conv.append({"role": role, "content": new_text.strip()})
            else:
                # 텍스트만 있는 경우 그대로 유지
                new_conv.append({"role": role, "content": content})
        
        ret_messages.append(new_conv)
        ret_audios.append(new_conv_audios)

    return {"messages": ret_messages, "audios": ret_audios}

# -------------------------------------------------------------------------
# [3] 로더 함수 (기존 Transform 낚아채기 기술 적용)
# -------------------------------------------------------------------------
# dataset_meta를 받아주거나, *args로 모든 추가 인자를 무시해야 함
def my_hijack_loader(dataset_id, dataset_meta=None, **kwargs):
    quiet_print("🚀 Loading Dataset via easy_load (Lazy Mode)...")
    
    # 1. easy_load 호출 (이 시점에는 오디오가 로드되지 않음)
    ds = easy_load(format="chat")
    # --------------------------------------------------------
    # [🔥 핵심 수정] 20개만 자르기 (데이터 로드 없이 인덱스만 자름)
    # --------------------------------------------------------
    TEST_COUNT = 20
    if len(ds) > TEST_COUNT:
        quiet_print(f"✂️ [TEST MODE] Slicing dataset: {len(ds)} -> {TEST_COUNT} samples.")
        # .select()는 데이터를 메모리에 올리지 않고 View만 만듭니다. (Lazy 유지)
        ds = ds.select(range(TEST_COUNT))
    else:
        quiet_print(f"ℹ️ Dataset is smaller than {TEST_COUNT}, using full dataset.")
    # --------------------------------------------------------
    # 2. [핵심 기술] 기존 Transform 함수 추출
    # easy_load가 설정해둔 '오디오 로딩 로직'을 가져옵니다.
    # (HuggingFace Dataset 내부 변수 _format_kwargs 접근)
    old_transform = ds._format_kwargs.get('transform')
    
    if old_transform is None:
        quiet_print("⚠️ Warning: No existing transform found. Making a generic one.")
        old_transform = lambda x: x

    # 3. 새로운 Transform 정의 (기존 로직 + 변환 로직 연결)
    def new_lazy_transform(batch):
        # (1) 먼저 easy_load의 기존 로직을 실행하여 오디오를 로드함 (Lazy)
        intermediate_batch = old_transform(batch)
        
        # (2) 로드된 데이터의 포맷을 Qwen용으로 변환함
        final_batch = convert_batch_to_qwen_format(intermediate_batch)
        
        return final_batch

    # 4. 데이터셋에 새로운 Transform 적용
    ds.set_transform(new_lazy_transform)
    
    quiet_print(f"✅ Transform Hijacked & Applied. Total samples: {len(ds)}")
    
    # 테스트용 슬라이싱 (필요 시)
    # if len(ds) > 10:
    #    ds = ds.select(range(10))
        
    return ds

# -------------------------------------------------------------------------
# [4] 데이터셋 등록
# -------------------------------------------------------------------------
register_dataset(
    DatasetMeta(
        dataset_name=DATASET_NAME,
        load_function=my_hijack_loader,
    )
)

# # -------------------------------------------------------------------------
# # [5] 학습 설정 (요청하신 대로 OOM 방지 옵션 제거)
# # -------------------------------------------------------------------------
# train_args = TrainArguments(
#     model_kwargs={"device_map": "auto"},
#     model=MODEL_ID,
#     model_type=None,

#     custom_register_path="./template.py",
#     template=CUSTOM_TEMPLATE,

#     dataset=[DATASET_NAME],

#     train_type="lora",
#     # [Target Modules] Thinker만 타겟팅 (성공했던 설정)
#     target_modules=r"^thinker\.model\.layers\.\d+\..*(q|k|v|o)_proj$",

#     lora_rank=16,
#     lora_alpha=32,
#     lora_dropout=0.05,

#     freeze_vit=True,
#     freeze_aligner=True,

#     bf16=True,
#     num_train_epochs=1,
#     per_device_train_batch_size=1,
#     gradient_accumulation_steps=4,
#     learning_rate=1e-4,
#     max_length=2048,
#     output_dir="./qwen3_omni_sca_result",

#     logging_steps=1,
#     save_steps=10,
#     save_total_limit=2,

#     # [Lazy 설정]
#     lazy_tokenize=True,
#     dataset_num_proc=1,      
#     dataloader_num_workers=0, # 메인 프로세스에서 로드 (오류 방지)
    
#     load_from_cache_file=False, 
# )
# ... (앞부분 동일) ...
train_args = TrainArguments(
    # --- 기본 설정 ---
    model=MODEL_ID,
    model_type="qwen3_omni",
    #model_type=None,
    custom_register_path="./template.py",
    template="qwen3-omni-sca",
    
    # 데이터셋 직접 경로 지정
    dataset=[DATASET_NAME],
    # --- 학습 방식 ---
    train_type="lora",
    
    # [Target Modules] 문제의 MLP 레이어 포함 (4bit 로드 시 메모리 문제 없음)
    target_modules=r"^thinker\.model\.layers\.\d+\..*(q|k|v|o)_proj$",

    # --- LoRA 설정 ---
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.05,

    # --- 멀티모달 동결 ---
    freeze_vit=True,
    freeze_aligner=True,

    # --- ★★★ [핵심 수정] Quantization Arguments (공식 문서 기준) ★★★ ---
    # 1. 양자화 방식 지정 (필수)
    quant_method="bnb", 
    
    # 2. 비트 수 (quantization_bit -> quant_bits)
    quant_bits=4, 
    
    # 3. 연산 타입 (bnb_4bit_comp_dtype -> bnb_4bit_compute_dtype)
    bnb_4bit_compute_dtype="bfloat16", 
    
    # 4. 양자화 타입
    bnb_4bit_quant_type="nf4",
    
    # 5. 이중 양자화 사용 (메모리 추가 절약)
    bnb_4bit_use_double_quant=True,

    # --- 학습 하이퍼파라미터 ---
    bf16=True,  # A40은 bf16 지원함
    num_train_epochs=1,
    per_device_train_batch_size=1,
    
    # Gradient Checkpointing은 4bit 학습 시 필수 (메모리 절약)
    gradient_checkpointing=True, 
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    max_length=2048,

    # --- 저장 및 로깅 ---
    output_dir="/workspace/qwen3_omni_sca_result",
    logging_steps=1,
    save_steps=10,
    save_total_limit=2,

    # --- 데이터 처리 ---
    lazy_tokenize=True,
    dataset_num_proc=1,      
    dataloader_num_workers=0,
    load_from_cache_file=False,

    #ddp 학습 위한 설정 

# # NPROC_PER_NODE=2 : GPU 2개를 쓰겠다는 뜻
# torchrun --nproc_per_node=2 --master_port=29500 train_fly_ver.py
    #optim="paged_adamw_8bit",
    ddp_find_unused_parameters=True,
    
    # --- 호환성 및 안전 장치 ---
    # check_dataset_strategy="none",  # 필요 시 주석 해제 (이전 오류 관련)
    # model_kwargs={"pad_token_id": 151645} # 필요 시 주석 해제 (이전 오류 관련)
)

# ... (뒷부분 동일) ...
# -------------------------------------------------------------------------
# [6] 학습 시작
# -------------------------------------------------------------------------
if __name__ == "__main__":
    quiet_print("🏁 Starting training (Integrated Lazy Mode)...")
    sft_main(train_args)