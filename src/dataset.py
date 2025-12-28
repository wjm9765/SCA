# src/dataset.py
import os
import numpy as np
from swift.llm import register_dataset, DatasetMeta
from sca_data.dataset_utils import easy_load
from swift.utils import get_logger

logger = get_logger()

def convert_batch_to_qwen_format(batch):
    """Standard Chat Format -> Qwen <audio> Tag Format"""
    ret_messages = []
    ret_audios = []

    for conversation in batch["messages"]:
        new_conv = []
        new_conv_audios = []
        for msg in conversation:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, list):
                new_text = ""
                for item in content:
                    if item["type"] == "text":
                        new_text += item["text"] + "\n"
                    elif item["type"] == "audio":
                        wave = item.get("audio_waveform")
                        if isinstance(wave, np.ndarray) and wave.dtype != np.float32:
                            wave = wave.astype(np.float32)
                        new_conv_audios.append(wave)
                        new_text += "<audio>"
                new_conv.append({"role": role, "content": new_text.strip()})
            else:
                new_conv.append({"role": role, "content": content})
        
        ret_messages.append(new_conv)
        ret_audios.append(new_conv_audios)

    return {"messages": ret_messages, "audios": ret_audios}

def my_hijack_loader(dataset_id, dataset_meta=None, **kwargs):
    logger.info("🚀 Loading Dataset via easy_load (Hijacked Mode)...")
    ds = easy_load(format="chat")
    
    # #테스트용 슬라이싱 데이터
    # TEST_COUNT = 20
    # if len(ds) > TEST_COUNT:
    #     logger.info(f"✂️ [TEST MODE] Slicing dataset: {len(ds)} -> {TEST_COUNT}")
    #     ds = ds.select(range(TEST_COUNT))

    old_transform = ds._format_kwargs.get('transform', lambda x: x)

    def new_lazy_transform(batch):
        intermediate_batch = old_transform(batch)
        final_batch = convert_batch_to_qwen_format(intermediate_batch)
        return final_batch

    ds.set_transform(new_lazy_transform)
    return ds

def register_custom_dataset(dataset_name: str):
    """데이터셋 등록 함수"""
    register_dataset(
        DatasetMeta(
            dataset_name=dataset_name,
            load_function=my_hijack_loader,
        )
    )
    logger.info(f"✅ Dataset '{dataset_name}' registered successfully.")