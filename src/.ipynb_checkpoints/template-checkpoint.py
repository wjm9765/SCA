# import torch
# import numpy as np
# from typing import Literal, List, Dict, Any

# from swift.llm import (
#     register_template, TemplateMeta, get_model_tokenizer, get_template
# )
# from swift.llm.template.template_inputs import StdTemplateInputs
# from swift.llm.template.utils import Context
# from swift.llm.template.vision_utils import load_audio
# from swift.utils import get_logger

# # -------------------------------------------------------------------------
# # [1] Qwen3 Omni 순정 템플릿 클래스 임포트
# # -------------------------------------------------------------------------
# # 당신이 확인한 경로에서 정확한 클래스를 가져옵니다.
# try:
#     from swift.llm.template.template.qwen import Qwen3OmniTemplate
# except ImportError:
#     # 경로가 다를 경우를 대비한 안전장치 (보통은 위 경로가 맞음)
#     try:
#         from swift.llm.template.template.qwen import Qwen2_5OmniTemplate as Qwen3OmniTemplate
#     except:
#         from swift.llm.template.base import Template as Qwen3OmniTemplate

# logger = get_logger()
# print(f"ℹ️ Base Template Class: {Qwen3OmniTemplate.__name__}")

# # -------------------------------------------------------------------------
# # [2] Custom Template Class (Inheritance & Override)
# # -------------------------------------------------------------------------
# class Qwen3OmniSCATemplate(Qwen3OmniTemplate):
#     """
#     Qwen3OmniTemplate을 상속받아 모든 기능을 유지하되,
#     1. Numpy 오디오 입력 시 파일 로딩 스킵
#     2. 정확한 Audio Placeholder (<|audio_pad|>) 사용
#     # """
    
#     # def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
#     #                 inputs: StdTemplateInputs) -> List[Context]:
        
#     #     # [Override] Audio 태그일 때만 개입
#     #     if media_type == "audio":
#     #         audio_data = inputs.audios[index]

#     #         # 1. 이미 메모리에 있는 데이터(Numpy/Tensor/List)라면? -> 로딩 스킵 (PASS)
#     #         if isinstance(audio_data, (np.ndarray, torch.Tensor, list)):
#     #             # 아무 작업도 안함. (Processor가 나중에 처리)
#     #             pass
            
#     #         # 2. 파일 경로(문자열)라면? -> 부모 클래스의 원래 로직(load_audio) 수행
#     #         elif isinstance(audio_data, str):
#     #             if self.mode != 'vllm':
#     #                 inputs.audios[index] = load_audio(audio_data, self.sampling_rate)
            
#     #         # [핵심 수정] Qwen3 Omni 소스코드에 명시된 placeholder 사용
#     #         # <|AUDIO|> 대신 <|audio_pad|>를 리턴해야 토크나이저 에러가 안 남
#     #         return ["<|audio_pad|>"]

#     #     # Image나 Video는 부모 클래스 로직 그대로 사용
#     #     return super().replace_tag(media_type, index, inputs)
    
#     def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
#                     inputs: StdTemplateInputs) -> List[Context]:
        
#         if media_type == "audio":
#             audio_data = inputs.audios[index]
    
#             # [1] Numpy 스킵 로직 (사용자님 로직 유지)
#             if isinstance(audio_data, (np.ndarray, torch.Tensor, list)):
#                 pass 
            
#             # [2] 파일 로딩 (부모 로직 활용)
#             elif isinstance(audio_data, str):
#                 if self.mode != 'vllm':
#                     inputs.audios[index] = load_audio(audio_data, self.sampling_rate)
            
#             # [3] 리턴값 수정 (오픈소스 소스코드 기준)
#             # Qwen3-Omni는 반드시 start와 end 토큰이 함께 있어야 합니다.
#             if self.version == 'omni_v3':
#                 return ['<|audio_start|><|audio_pad|><|audio_end|>']
#             else:
#                 # 혹시 모를 하위 호환성 (omni_v2_5)
#                 return ['<|audio_bos|><|AUDIO|><|audio_eos|>']
    
#         return super().replace_tag(media_type, index, inputs)
# # -------------------------------------------------------------------------
# # [3] 템플릿 등록
# # -------------------------------------------------------------------------
# TEMPLATE_NAME = "qwen3-omni-sca"

# register_template(
#     TemplateMeta(
#         TEMPLATE_NAME,
#         # Qwen3 Omni의 Chat Format에 맞춤
#         prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
#         prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
#         chat_sep=['<|im_end|>\n'],
#         suffix=['<|im_end|>'],
#         default_system="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
#         template_cls=Qwen3OmniSCATemplate # 우리가 만든 클래스 지정
#     )
# )

# print(f"✅ Template '{TEMPLATE_NAME}' registered successfully.")


# # -------------------------------------------------------------------------
# # [4] Main Debugging Block (데이터 구조 및 토큰 ID 검증)
# # -------------------------------------------------------------------------
# if __name__ == "__main__":
#     print("\n🚀 Starting Template Debugging...")

#     # 1. 모델 & 프로세서 로드 (실제 경로 사용)
#     model_id = "/workspace/modelscope_cache/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"
    
#     try:
#         print(f"📥 Loading Processor from: {model_id}")
#         # load_model=False로 가볍게 로드
#         _, tokenizer = get_model_tokenizer(model_id, load_model=False)
        
#         # Processor 별도 로드 (Qwen3OmniProcessor)
#         from transformers import AutoProcessor
#         processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

#         # 2. 등록한 커스텀 템플릿 가져오기
#         template = get_template(TEMPLATE_NAME, tokenizer)
#         template.processor = processor # 프로세서 주입 필수
        
#         # 3. [Test Case] 16000Hz 실수형 Numpy 오디오 생성
#         sr = 16000
#         duration = 1.0 # 1초
#         dummy_audio_numpy = np.random.uniform(-1.0, 1.0, int(sr * duration)).astype(np.float32)
        
#         print(f"🔊 Generated Dummy Audio: Shape={dummy_audio_numpy.shape}")

#         # 4. 입력 데이터 구성
#         # <audio> 태그는 템플릿의 replace_tag를 트리거함
#         # replace_tag가 <|audio_pad|>를 반환하면, 템플릿은 이를 151675번 토큰으로 변환해야 함
#         input_data = {
#             "messages": [
#                 {"role": "system", "content": "System Prompt"},
#                 {"role": "user", "content": "<audio>\nTest audio."}, 
#                 {"role": "assistant", "content": "Response"}
#             ],
#             "audios": [dummy_audio_numpy] 
#         }

#         # 5. 인코딩 실행
#         print("⚙️ Encoding data via Template...")
#         template.set_mode('train') 
#         encoded = template.encode(input_data)
        
#         # 6. 결과 정밀 검증
#         print("\n✅ Encode Success!")
#         input_ids = encoded['input_ids']
        
#         # (1) 오디오 토큰 ID 확인
#         # Config에 명시된 151675가 input_ids에 포함되어 있는지 확인
#         target_audio_id = 151675
#         if target_audio_id in input_ids:
#             print(f"🎯 [SUCCESS] Audio Token ID ({target_audio_id}) found in input_ids!")
#             count = input_ids.count(target_audio_id)
#             print(f"   -> Count: {count} (Should be proportional to audio length)")
#         else:
#             print(f"🚨 [FAIL] Audio Token ID ({target_audio_id}) NOT found in input_ids.")
#             print(f"   -> First 20 tokens: {input_ids[:20]}")
            
#         # (2) 오디오 피처 확인
#         if 'input_features' in encoded:
#             print(f"🎵 Audio Features Shape: {encoded['input_features'].shape}")
#         else:
#             # Qwen3 Omni 프로세서는 input_features 대신 다른 키를 쓸 수도 있음 (예: pixel_values_audio 등)
#             # encoded 키 전체 출력해서 확인
#             print(f"ℹ️ All Encoded Keys: {list(encoded.keys())}")

#     except Exception as e:
#         print(f"\n🚨 Debugging Failed: {e}")
#         import traceback
#         traceback.print_exc()


# src/template.py
import torch
import numpy as np
from typing import Literal, List
from swift.llm import register_template, TemplateMeta, get_template
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.llm.template.utils import Context
from swift.llm.template.vision_utils import load_audio
from swift.utils import get_logger

# Import Base Template
try:
    from swift.llm.template.template.qwen import Qwen3OmniTemplate
except ImportError:
    from swift.llm.template.base import Template as Qwen3OmniTemplate

logger = get_logger()

class Qwen3OmniSCATemplate(Qwen3OmniTemplate):
    """Custom Template for SCA Project"""
    def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                    inputs: StdTemplateInputs) -> List[Context]:
        if media_type == "audio":
            audio_data = inputs.audios[index]
            # 1. Numpy/Tensor Skip Logic
            if isinstance(audio_data, (np.ndarray, torch.Tensor, list)):
                pass 
            # 2. File Path Logic
            elif isinstance(audio_data, str):
                if self.mode != 'vllm':
                    inputs.audios[index] = load_audio(audio_data, self.sampling_rate)
            
            # 3. Return Correct Placeholder
            if getattr(self, 'version', '') == 'omni_v3':
                return ['<|audio_start|><|audio_pad|><|audio_end|>']
            else:
                return ['<|audio_bos|><|AUDIO|><|audio_eos|>']
        
        return super().replace_tag(media_type, index, inputs)

def register_custom_template():
    """외부에서 호출 시 템플릿을 등록하는 함수"""
    TEMPLATE_NAME = "qwen3-omni-sca"
    register_template(
        TemplateMeta(
            TEMPLATE_NAME,
            prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
            prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
            chat_sep=['<|im_end|>\n'],
            suffix=['<|im_end|>'],
            default_system="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            template_cls=Qwen3OmniSCATemplate
        )
    )
    logger.info(f"✅ Template '{TEMPLATE_NAME}' registered successfully.")