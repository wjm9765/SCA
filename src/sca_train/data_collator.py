import numpy as np
import torch
from transformers import Qwen3OmniMoeProcessor

# Expected speaker embedding dimension (ECAPA-TDNN output)
SPEAKER_EMBEDDING_DIM = 192


class Qwen3OmniCollator:
    def __init__(
        self,
        processor: Qwen3OmniMoeProcessor,
        mask_instruction: bool = True,
        max_length: int = 32768,
        train_talker: bool = False,
    ):
        self.processor = processor
        self.mask_instruction = mask_instruction
        self.max_length = max_length
        self.train_talker = train_talker

        assert hasattr(processor, "tokenizer"), (
            "Processor must have a tokenizer attribute"
        )
        self.pad_token_id = processor.tokenizer.pad_token_id
        self.im_start_id = None

        im_tokens = processor.tokenizer.encode("<|im_start|>", add_special_tokens=False)
        if len(im_tokens) > 0:
            self.im_start_id = im_tokens[0]

    def __call__(self, features):
        # 1. Handle Input Format
        if isinstance(features, dict):
            keys = features.keys()
            features = [dict(zip(keys, vals)) for vals in zip(*features.values())]

        texts = []
        audios = []
        assistant_audios = []
        speaker_embeddings = []

        # 2. Extract Data
        for feature in features:
            messages = feature["messages"]
            feature_audio = None
            assistant_audio = None
            speaker_embedding = None
            user_found = False
            assistant_found = False

            for msg in messages:
                if msg["role"] == "user":
                    if user_found:
                        raise ValueError(
                            "Multiple user messages found in a single input."
                        )
                    user_found = True
                    for content in msg["content"]:
                        if content["type"] == "audio":
                            if "audio_waveform" in content:
                                if feature_audio is not None:
                                    raise ValueError(
                                        "Multiple user audio contents found in a single input."
                                    )
                                if content["sampling_rate"] != 16000:
                                    raise ValueError(
                                        "User audio sampling rate must be 16000 Hz."
                                    )
                                feature_audio = content.pop("audio_waveform")
                                content["audio_url"] = "place_holder"

                if msg["role"] == "assistant":
                    if assistant_found:
                        raise ValueError(
                            "Multiple assistant messages found in a single input."
                        )
                    if not user_found:
                        raise ValueError("Assistant message found before user message.")
                    assistant_found = True

                    remove_idx = []
                    for i, content in enumerate(msg["content"]):
                        if content["type"] == "audio":
                            if "audio_waveform" in content:
                                if assistant_audio is not None:
                                    raise ValueError(
                                        "Multiple assistant audio contents found in a single input."
                                    )
                                if content["sampling_rate"] != 24000:
                                    raise ValueError(
                                        "Assistant audio sampling rate must be 24000 Hz."
                                    )
                                assistant_audio = content.pop("audio_waveform")
                                assistant_audios.append(assistant_audio)
                                remove_idx.append(i)

                            # Extract pre-computed speaker embedding
                            if "speaker_embedding" in content:
                                if speaker_embedding is not None:
                                    raise ValueError(
                                        "Multiple speaker embeddings found in a single input."
                                    )
                                speaker_embedding = content.pop("speaker_embedding")

                                # Convert to tensor (handles list, numpy array, or tensor)
                                if isinstance(speaker_embedding, list):
                                    speaker_embedding = torch.tensor(
                                        speaker_embedding, dtype=torch.float32
                                    )
                                elif isinstance(speaker_embedding, np.ndarray):
                                    speaker_embedding = torch.from_numpy(
                                        speaker_embedding
                                    )

                                if speaker_embedding.ndim == 1:
                                    speaker_embedding = speaker_embedding.unsqueeze(
                                        0
                                    )  # [192] -> [1, 192]

                                assert speaker_embedding.shape == (
                                    1,
                                    SPEAKER_EMBEDDING_DIM,
                                ), (
                                    f"Speaker embedding must be shape [1, {SPEAKER_EMBEDDING_DIM}], "
                                    f"got {speaker_embedding.shape}"
                                )
                                speaker_embeddings.append(speaker_embedding)

                    for idx in reversed(remove_idx):
                        del msg["content"][idx]

            if feature_audio is None:
                feature_audio = np.zeros(16000)
            if self.train_talker ^ (assistant_audio is not None):
                raise ValueError(
                    "Mismatch in talker training mode and presence of assistant audio."
                )

            # Validate speaker embedding is present when training talker
            if self.train_talker and speaker_embedding is None:
                raise ValueError(
                    "speaker_embedding must be provided in assistant audio content when train_talker=True. "
                    "Pre-compute speaker embeddings in your dataset using ECAPA-TDNN."
                )

            audios.append(feature_audio)

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

        # 3. Process Batch
        batch = self.processor(
            text=texts,
            audio=audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # 4. Create Labels
        labels = batch["input_ids"].clone()

        # Apply Padding Mask
        if self.pad_token_id is not None:
            labels[labels == self.pad_token_id] = -100

        # Apply Instruction Masking
        # Data always ends with assistant response and is single-turn
        if self.mask_instruction and self.im_start_id is not None:
            for i in range(len(batch["input_ids"])):
                starts = (batch["input_ids"][i] == self.im_start_id).nonzero(
                    as_tuple=True
                )[0]
                if len(starts) > 0:
                    last_start = starts[-1]
                    header_offset = 3
                    labels[i, : last_start + header_offset] = -100

                # system, user, assistant -> 3 starts max
                if len(starts) > 3:
                    raise ValueError(
                        "Multiple <|im_start|> tokens found in a single-turn input."
                    )

        batch["labels"] = labels
        if len(assistant_audios) > 0:
            batch["assistant_audios"] = assistant_audios

        # Stack speaker embeddings into a batch tensor [batch, 192]
        if len(speaker_embeddings) > 0:
            batch["speaker_embeddings"] = torch.cat(speaker_embeddings, dim=0).float()
            assert batch["speaker_embeddings"].shape == (
                len(speaker_embeddings),
                SPEAKER_EMBEDDING_DIM,
            ), (
                f"Batched speaker_embeddings shape mismatch: expected "
                f"[{len(speaker_embeddings)}, {SPEAKER_EMBEDDING_DIM}], got {batch['speaker_embeddings'].shape}"
            )

        return batch
