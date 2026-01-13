from typing import List, Optional, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Qwen3OmniMoeForConditionalGeneration,
    AutoFeatureExtractor,
    MimiModel,
    Qwen3OmniMoeConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast


class Qwen3OmniMoeWithProperForwardConfig(Qwen3OmniMoeConfig):
    model_type = "qwen3_omni_moe_with_proper_forward"

    def __init__(self, train_mtp: bool = False, **kwargs):
        self.train_mtp = train_mtp
        super().__init__(**kwargs)


class Qwen3OmniMoeWithProperForward(Qwen3OmniMoeForConditionalGeneration):
    config_class = Qwen3OmniMoeWithProperForwardConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # NOTE: mimi_model and mimi_feature_extractor are intentionally NOT loaded here.
        # They must be loaded AFTER from_pretrained() completes to avoid weight corruption.
        # Call load_mimi_model() after instantiation to load them properly.
        self.mimi_model = None
        self.mimi_feature_extractor = None

        # Projection layer: speaker embedding (192 dim from ECAPA-TDNN) -> talker hidden size
        # Speaker embeddings are pre-computed in the dataset, not extracted here
        # This layer is kept in fp16/bf16 (not quantized) via llm_int8_skip_modules in train.py
        speaker_embed_dim = 192  # ECAPA-TDNN output dimension
        talker_hidden_size = self.config.talker_config.text_config.hidden_size
        self.speaker_projection = nn.Linear(speaker_embed_dim, talker_hidden_size)

    def load_mimi_model(self) -> None:
        """Load Mimi model and feature extractor.

        IMPORTANT: This must be called AFTER from_pretrained() to avoid weight corruption.
        The Mimi model is loaded separately because it's not part of the main checkpoint,
        and loading it in __init__ causes it to be reinitialized with garbage weights.
        """
        if self.mimi_model is not None:
            print("Warning: Mimi model already loaded, skipping...")
            return

        print("Loading Mimi model from kyutai/mimi...")
        self.mimi_feature_extractor = AutoFeatureExtractor.from_pretrained(
            "kyutai/mimi"
        )
        self.mimi_model = MimiModel.from_pretrained(
            "kyutai/mimi",
            torch_dtype=self.talker.dtype,
        )

        self.mimi_model.eval()
        for param in self.mimi_model.parameters():
            param.requires_grad = False

        print("Mimi model loaded successfully!")

    def _get_unwrapped_code_predictor(self):
        """Get the code_predictor unwrapped from PEFT's ModulesToSaveWrapper if present.

        PEFT wraps modules in `modules_to_save` with ModulesToSaveWrapper, which has a
        forward(x, *args, **kwargs) signature that requires positional arg `x`. This method
        returns the underlying module so we can call it with keyword arguments.
        """
        code_predictor = self.talker.code_predictor

        if hasattr(code_predictor, "modules_to_save"):
            # Get the first active adapter's module
            active_adapters = getattr(code_predictor, "active_adapters", ["thinker"])
            if active_adapters and active_adapters[0] in code_predictor.modules_to_save:
                code_predictor = code_predictor.modules_to_save[active_adapters[0]]
            elif "thinker" in code_predictor.modules_to_save:
                code_predictor = code_predictor.modules_to_save["thinker"]
            else:
                # Fallback to original module
                code_predictor = code_predictor.original_module

        return code_predictor

    def _get_talker_assistant_parts(
        self,
        im_start_index,
        segment_end_index,
        speaker_embedding: torch.Tensor,
        thinker_embed,
        tts_pad_embed,
        tts_bos_embed,
        tts_eos_embed,
    ):
        """Build talker assistant parts with speaker embedding instead of speaker_id.

        This overrides the parent class method to use a continuous speaker embedding
        (extracted from audio via ECAPA-TDNN) instead of a discrete speaker token ID.
        This enables voice cloning with unlimited speakers.

        Args:
            im_start_index: Start index of assistant segment
            segment_end_index: End index of assistant segment
            speaker_embedding: Speaker embedding tensor [1, 192] from ECAPA-TDNN
            thinker_embed: Thinker embeddings
            tts_pad_embed: TTS pad embedding
            tts_bos_embed: TTS BOS embedding
            tts_eos_embed: TTS EOS embedding

        Returns:
            Tuple of (input_embeds, input_ids, trailing_text_hidden)
        """
        # Project speaker embedding to talker hidden size
        projected_speaker = self.speaker_projection(
            speaker_embedding
        )  # [1, talker_hidden_size]

        assistant_hidden = self.talker.text_projection(
            thinker_embed[:, im_start_index:segment_end_index]
        ).to(self.talker.device)  # [1, seq_len, hidden]

        assistant_text_hidden = torch.cat(
            (
                assistant_hidden[:, :3],
                tts_pad_embed.expand(-1, 4, -1),
                tts_bos_embed,
                assistant_hidden[:, 3:4],  # First text token
            ),
            dim=1,
        )

        # Embed codec special tokens (using placeholder for speaker slot)
        codec_special_tokens = torch.tensor(
            [
                [
                    self.config.talker_config.codec_nothink_id,
                    self.config.talker_config.codec_think_bos_id,
                    self.config.talker_config.codec_think_eos_id,
                    self.config.talker_config.codec_pad_id,  # Placeholder - will be replaced
                    self.config.talker_config.codec_pad_id,
                    self.config.talker_config.codec_bos_id,
                ]
            ],
            device=self.talker.device,
            dtype=torch.long,
        )

        codec_embeds_raw = self.talker.get_input_embeddings()(codec_special_tokens).to(
            self.talker.device
        )

        # Replace position 3 (speaker slot) with projected speaker embedding
        # Use torch.cat instead of in-place assignment to avoid autograd issues
        codec_embeds = torch.cat(
            [
                codec_embeds_raw[:, :3, :],  # positions 0, 1, 2
                projected_speaker.to(codec_embeds_raw.dtype).unsqueeze(
                    1
                ),  # position 3 (speaker)
                codec_embeds_raw[:, 4:, :],  # positions 4, 5
            ],
            dim=1,
        )

        assistant_codec_hidden = torch.cat(
            (
                torch.zeros(
                    (1, 3, self.config.talker_config.text_config.hidden_size),
                    device=self.talker.device,
                    dtype=self.talker.dtype,
                ),
                codec_embeds,
            ),
            dim=1,
        )

        trailing_text_hidden = torch.cat(
            (
                assistant_hidden[:, 4:],
                tts_eos_embed,
            ),
            dim=1,
        )

        input_embeds = assistant_text_hidden + assistant_codec_hidden
        input_ids = torch.full(
            (1, assistant_text_hidden.shape[1]),
            fill_value=self.config.tts_pad_token_id,
            dtype=torch.long,
            device=assistant_text_hidden.device,
        )

        return input_embeds, input_ids, trailing_text_hidden

    def _encode_audio_to_codes(self, audios: List[np.ndarray]) -> torch.Tensor:
        """
        Encode assistant audio to Mimi codes.

        NOTE: Expects audios to be at 24kHz (Mimi's native sampling rate).
        User audios for Thinker remain at 16kHz (Qwen-Audio standard).
        """
        # Ensure mimi_model is on the correct device (lazy move)
        # Use next(parameters()).device to get actual device
        mimi_device = next(self.mimi_model.parameters()).device
        if mimi_device != self.device:
            self.mimi_model = self.mimi_model.to(self.device)  # type: ignore[arg-type]

        processed_audios = []
        for audio in audios:
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            processed_audios.append(audio)

        # Use Mimi's native 24kHz sampling rate
        audio_inputs = self.mimi_feature_extractor(
            processed_audios, sampling_rate=24000, return_tensors="pt", padding=True
        )

        audio_tensor = audio_inputs["input_values"].to(
            device=self.device,
            dtype=self.mimi_model.dtype,
        )

        if audio_tensor.ndim == 2:
            audio_tensor = audio_tensor.unsqueeze(1)
        elif audio_tensor.ndim != 3:
            raise ValueError(f"Unexpected Mimi input shape: {audio_tensor.shape}")

        with torch.no_grad():
            output = self.mimi_model.encode(audio_tensor)
            # mimi_model.encode returns a tuple (codes, something_else) or object with audio_codes
            # Based on diagnostics, it seems to return tuple[Tensor, Tensor | None]
            if isinstance(output, tuple):
                codes = output[0]
            elif hasattr(output, "audio_codes"):
                codes = output.audio_codes
            else:
                codes = output

        if codes is None:
            raise ValueError("Mimi model returned None codes")

        # Ensure result is a Tensor
        if not isinstance(codes, torch.Tensor):
            # Try to access .audio_codes if it's an output object
            if hasattr(codes, "audio_codes"):
                codes = codes.audio_codes
            else:
                raise ValueError(f"Expected Tensor from mimi encode, got {type(codes)}")

        # Final explicit check to satisfy type checker
        if not isinstance(codes, torch.Tensor):
            raise ValueError("Failed to obtain Tensor codes")

        return codes

    def _align_codebook_dim(self, codes: torch.Tensor) -> torch.Tensor:
        target_quantizers = self.code2wav.config.num_quantizers
        assert target_quantizers is not None, "code2wav.config.num_quantizers is None"

        current = codes.shape[1]
        if current == target_quantizers:
            return codes
        if current < target_quantizers:
            raise ValueError(
                f"Mimi codes have {current} quantizers but model expects {target_quantizers}."
            )
        return codes[:, :target_quantizers, :]

    def _encode_and_align_audio(self, audio: List[np.ndarray]) -> torch.Tensor:
        codes = self._encode_audio_to_codes(audio).long()
        aligned_codes = self._align_codebook_dim(codes)
        return aligned_codes

    def _build_talker_prefix_single(
        self,
        thinker_outputs,
        input_ids: torch.Tensor,
        speaker_embedding: torch.Tensor,
        batch_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build talker prefix for a single sample in the batch.

        Args:
            thinker_outputs: Output from thinker model with hidden states
            input_ids: Input token IDs [batch, seq_len]
            speaker_embedding: Speaker embedding [1, 192] from ECAPA-TDNN
            batch_idx: Index of sample in batch

        Returns:
            Tuple of (talker_input_embed, talker_input_id, trailing_text_hidden, tts_pad_embed)
        """
        config = self.config
        thinker_embed = thinker_outputs.hidden_states[0][batch_idx : batch_idx + 1].to(
            self.device
        )
        accept_layer = config.talker_config.accept_hidden_layer
        thinker_hidden = thinker_outputs.hidden_states[accept_layer][
            batch_idx : batch_idx + 1
        ].to(self.device)

        # Get im_start positions for this sample
        sample_input_ids = input_ids[batch_idx]
        im_start_positions = torch.nonzero(
            sample_input_ids == config.im_start_token_id
        ).view(-1)
        im_start_indexes = torch.cat(
            (
                im_start_positions,
                torch.tensor(
                    [input_ids.shape[1]], device=self.device, dtype=input_ids.dtype
                ),
            ),
            dim=0,
        )

        # Multimodal mask for this sample (keep batch dim for compatibility)
        sample_multimodal_mask = (
            (
                input_ids[batch_idx : batch_idx + 1]
                == config.thinker_config.audio_token_id
            )
            | (
                input_ids[batch_idx : batch_idx + 1]
                == config.thinker_config.image_token_id
            )
            | (
                input_ids[batch_idx : batch_idx + 1]
                == config.thinker_config.video_token_id
            )
        ).to(self.device)

        talker_special_tokens = torch.tensor(
            [
                [
                    config.tts_bos_token_id,
                    config.tts_eos_token_id,
                    config.tts_pad_token_id,
                ]
            ],
            device=self.device,
            dtype=input_ids.dtype,
        )
        # Handle LoRA-wrapped thinker model
        thinker_embeddings = self.thinker.get_input_embeddings()
        if hasattr(thinker_embeddings, "base_layer"):
            thinker_embeddings = thinker_embeddings.base_layer

        # Ensure thinker_embeddings is callable (it should be nn.Embedding)
        if not callable(thinker_embeddings):
            # Fallback if it's just weights
            embedding_weights = thinker_embeddings
            thinker_embeddings = lambda x: F.embedding(x, embedding_weights)

        tts_bos_embed, tts_eos_embed, tts_pad_embed = (
            self.talker.text_projection(thinker_embeddings(talker_special_tokens))
            .to(self.device)
            .chunk(3, dim=1)
        )

        talker_input_embeds, talker_input_ids = [], []
        trailing_text_hidden = None

        for i in range(len(im_start_indexes) - 1):
            im_start_index = im_start_indexes[i]
            segment_end_index = im_start_indexes[i + 1]
            role_token = sample_input_ids[im_start_index + 1]
            if role_token == config.user_token_id:
                user_part = self._get_talker_user_parts(
                    im_start_index,
                    segment_end_index,
                    sample_multimodal_mask,
                    thinker_hidden,
                    thinker_embed,
                )
                talker_input_embeds.append(user_part)
                talker_input_ids.append(
                    input_ids[
                        batch_idx : batch_idx + 1, im_start_index:segment_end_index
                    ]
                )
            elif (
                role_token == config.assistant_token_id
                and i == len(im_start_indexes) - 2
            ):
                # Use speaker_embedding instead of speaker_id
                assistant_embeds, assistant_ids, trailing_text_hidden = (
                    self._get_talker_assistant_parts(
                        im_start_index,
                        segment_end_index,
                        speaker_embedding,  # Speaker embedding from ECAPA-TDNN
                        thinker_embed,
                        tts_pad_embed,
                        tts_bos_embed,
                        tts_eos_embed,
                    )
                )
                talker_input_embeds.append(assistant_embeds)
                talker_input_ids.append(assistant_ids)

        if trailing_text_hidden is None:
            raise RuntimeError(
                "Failed to build trailing_text_hidden for talker training."
            )

        talker_input_embed = torch.cat(
            [embed.to(self.device) for embed in talker_input_embeds], dim=1
        )
        talker_input_id = torch.cat(
            [ids.to(self.device) for ids in talker_input_ids], dim=1
        )
        return (
            talker_input_embed,
            talker_input_id,
            trailing_text_hidden.to(self.device),
            tts_pad_embed,
        )

    def _build_talker_prefix(
        self,
        thinker_outputs,
        input_ids: torch.Tensor,
        speaker_embeddings: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Build talker prefix for batch of samples.

        Args:
            thinker_outputs: Output from thinker model with hidden states
            input_ids: Input token IDs [batch, seq_len]
            speaker_embeddings: Speaker embeddings [batch, 192] from ECAPA-TDNN

        Returns:
            talker_input_embed: [batch, max_prefix_len, hidden]
            talker_input_id: [batch, max_prefix_len]
            trailing_text_hidden: [batch, max_trailing_len, hidden]
            tts_pad_embed: [1, 1, hidden]
            original_prefix_lens: [batch] - original prefix length per sample
            original_trailing_lens: [batch] - original trailing length per sample
        """
        batch_size = input_ids.shape[0]

        if batch_size == 1:
            # Fast path for batch_size=1
            talker_embed, talker_id, trailing_hidden, tts_pad = (
                self._build_talker_prefix_single(
                    thinker_outputs, input_ids, speaker_embeddings[0:1], batch_idx=0
                )
            )
            original_prefix_lens = torch.tensor(
                [talker_embed.shape[1]], device=self.device
            )
            original_trailing_lens = torch.tensor(
                [trailing_hidden.shape[1]], device=self.device
            )
            return (
                talker_embed,
                talker_id,
                trailing_hidden,
                tts_pad,
                original_prefix_lens,
                original_trailing_lens,
            )

        # Process each sample in batch
        batch_talker_embeds = []
        batch_talker_ids = []
        batch_trailing_hidden = []
        batch_tts_pad_embed = []
        original_prefix_lens = []
        original_trailing_lens = []

        for batch_idx in range(batch_size):
            # Extract speaker embedding for this sample [1, 192]
            sample_speaker_embedding = speaker_embeddings[batch_idx : batch_idx + 1]
            talker_embed, talker_id, trailing_hidden, tts_pad = (
                self._build_talker_prefix_single(
                    thinker_outputs, input_ids, sample_speaker_embedding, batch_idx
                )
            )
            batch_talker_embeds.append(talker_embed)
            batch_talker_ids.append(talker_id)
            batch_trailing_hidden.append(trailing_hidden)
            batch_tts_pad_embed.append(tts_pad)
            original_prefix_lens.append(talker_embed.shape[1])
            original_trailing_lens.append(trailing_hidden.shape[1])

        # Convert to tensors
        original_prefix_lens = torch.tensor(original_prefix_lens, device=self.device)
        original_trailing_lens = torch.tensor(
            original_trailing_lens, device=self.device
        )

        # Stack results - need to pad to same length
        # Find max prefix length in batch
        max_prefix_len = max(embed.shape[1] for embed in batch_talker_embeds)
        max_trailing_len = max(hidden.shape[1] for hidden in batch_trailing_hidden)

        # Pad each sample to max length
        padded_embeds = []
        padded_ids = []
        padded_trailing = []

        for idx in range(batch_size):
            embed = batch_talker_embeds[idx]
            ids = batch_talker_ids[idx]
            trailing = batch_trailing_hidden[idx]

            # Pad prefix embeddings
            if embed.shape[1] < max_prefix_len:
                pad_len = max_prefix_len - embed.shape[1]
                # Pad with zeros (will be masked by attention mask)
                embed_pad = torch.zeros(
                    1, pad_len, embed.shape[2], dtype=embed.dtype, device=embed.device
                )
                embed = torch.cat([embed, embed_pad], dim=1)
            padded_embeds.append(embed)

            # Pad prefix ids
            if ids.shape[1] < max_prefix_len:
                pad_len = max_prefix_len - ids.shape[1]
                # Pad with pad token
                ids_pad = torch.full(
                    (1, pad_len),
                    self.config.tts_pad_token_id,
                    dtype=ids.dtype,
                    device=ids.device,
                )
                ids = torch.cat([ids, ids_pad], dim=1)
            padded_ids.append(ids)

            # Pad trailing hidden
            if trailing.shape[1] < max_trailing_len:
                pad_len = max_trailing_len - trailing.shape[1]
                trailing_pad = torch.zeros(
                    1,
                    pad_len,
                    trailing.shape[2],
                    dtype=trailing.dtype,
                    device=trailing.device,
                )
                trailing = torch.cat([trailing, trailing_pad], dim=1)
            padded_trailing.append(trailing)

        talker_input_embed = torch.cat(padded_embeds, dim=0)
        talker_input_id = torch.cat(padded_ids, dim=0)
        trailing_text_hidden = torch.cat(padded_trailing, dim=0)
        # tts_pad_embed is the same for all samples, just use first one
        tts_pad_embed = batch_tts_pad_embed[0]

        return (
            talker_input_embed,
            talker_input_id,
            trailing_text_hidden,
            tts_pad_embed,
            original_prefix_lens,
            original_trailing_lens,
        )

    def forward(self, **kwargs):
        # Compute audio codebook (Talker inference target)
        # Target code is ground truth code for talker prediction
        target_codes = None
        speaker_embeddings = None
        if "assistant_audios" in kwargs:
            assistant_audios = kwargs.pop("assistant_audios")
            target_codes = self._encode_and_align_audio(assistant_audios)
            if target_codes.shape[0] != kwargs["input_ids"].shape[0]:
                raise ValueError(
                    "Batch size of assistant_audios does not match input_ids."
                )

        # Get pre-computed speaker embeddings from kwargs (computed in dataset/collator)
        if "speaker_embeddings" in kwargs:
            speaker_embeddings = kwargs.pop("speaker_embeddings")
            # Validate speaker embeddings shape
            if speaker_embeddings.ndim != 2 or speaker_embeddings.shape[1] != 192:
                raise ValueError(
                    f"speaker_embeddings must be [batch, 192], got {speaker_embeddings.shape}"
                )
            speaker_embeddings = speaker_embeddings.to(self.device)

        # Run thinker inference
        # Ensure output_hidden_states=True for talker prefix construction
        kwargs["output_hidden_states"] = True
        thinker_outputs = self.thinker(**kwargs)
        thinker_loss = (
            thinker_outputs.loss if hasattr(thinker_outputs, "loss") else None
        )
        if "labels" in kwargs and (thinker_loss is not None):
            # Both present is fine (training)
            pass
        elif "labels" not in kwargs and (thinker_loss is None):
            # Both missing is fine (inference)
            pass
        else:
            # One present, one missing -> Error
            raise ValueError(
                "Labels provided but thinker did not return loss, or vice versa."
            )

        # ========================================
        # Talker & MTP Training (only when target_codes available)
        # ========================================
        if target_codes is None:
            # Text-only training: only Thinker loss
            talker_loss = torch.tensor(0.0, device=self.device)
            mtp_avg_loss = torch.tensor(0.0, device=self.device)
        else:
            # Speaker embeddings are required for talker training
            if speaker_embeddings is None:
                raise ValueError(
                    "speaker_embeddings must be provided when assistant_audios is present"
                )

            # Build Talker prefix with speaker embeddings
            (
                talker_input_embed,
                talker_input_ids,
                trailing_text_hidden,
                tts_pad_embed,
                original_prefix_lens,
                original_trailing_lens,
            ) = self._build_talker_prefix(
                thinker_outputs=thinker_outputs,
                input_ids=kwargs["input_ids"],
                speaker_embeddings=speaker_embeddings,  # Pre-computed speaker embeddings [batch, 192]
            )

            # Talker gets layer 0 codebook as target
            layer0_codes = target_codes[
                :, 0, :
            ]  # [batch, num_tokens] (select layer 0 code)
            num_codec_tokens = layer0_codes.shape[1]

            # Assertions for shape validation
            batch_size = target_codes.shape[0]
            assert target_codes.ndim == 3, (
                f"Expected target_codes to be 3D [batch, layers, tokens], got {target_codes.shape}"
            )
            assert target_codes.shape[1] == 16, (
                f"Expected 16 codec layers, got {target_codes.shape[1]}"
            )
            assert layer0_codes.shape == (batch_size, num_codec_tokens), (
                f"layer0_codes shape mismatch: {layer0_codes.shape}"
            )

            # get embeddings
            layer0_embeddings = self.talker.get_input_embeddings()(
                layer0_codes.to(self.device)
            )
            # Use helper to unwrap from PEFT wrapper if present
            unwrapped_code_predictor = self._get_unwrapped_code_predictor()
            predictor_embeds = unwrapped_code_predictor.get_input_embeddings()

            # Assert embeddings shape
            assert layer0_embeddings.ndim == 3, (
                f"layer0_embeddings should be 3D, got {layer0_embeddings.shape}"
            )

            # ========================================
            # Talker Input Construction (Teacher Forcing) - BATCH PROCESSING
            # ========================================
            # For each sample in batch, we need to use its ORIGINAL trailing_text_hidden length
            # to decide when to use trailing_text_hidden vs tts_pad_embed

            # Sum embeddings from all layers for each position
            all_layer_embeds_sum = layer0_embeddings.clone()
            for j in range(len(predictor_embeds)):
                layer_j_codes = target_codes[:, j + 1, :]
                emb = predictor_embeds[j](layer_j_codes.to(self.device))
                all_layer_embeds_sum = all_layer_embeds_sum + emb

            # Build shifted input for codec tokens (teacher forcing)
            # Process per-sample to correctly handle variable trailing lengths
            batch_codec_inputs = []

            # Assert all_layer_embeds_sum shape
            assert all_layer_embeds_sum.shape == (
                batch_size,
                num_codec_tokens,
                layer0_embeddings.shape[2],
            ), f"all_layer_embeds_sum shape mismatch: {all_layer_embeds_sum.shape}"

            for batch_idx in range(batch_size):
                sample_trailing_len = original_trailing_lens[batch_idx].item()
                sample_trailing_hidden = trailing_text_hidden[
                    batch_idx : batch_idx + 1
                ]  # [1, max_trailing_len, hidden]
                sample_all_layer_sum = all_layer_embeds_sum[
                    batch_idx : batch_idx + 1
                ]  # [1, num_tokens, hidden]

                codec_input_embeds_list = []
                for pos in range(num_codec_tokens):
                    if pos == 0:
                        continue

                    prev_pos = pos - 1
                    # Use ORIGINAL length to decide
                    if prev_pos < sample_trailing_len:
                        text_hidden = sample_trailing_hidden[
                            :, prev_pos : prev_pos + 1, :
                        ]
                    else:
                        text_hidden = tts_pad_embed

                    pos_embed = (
                        sample_all_layer_sum[:, prev_pos : prev_pos + 1, :]
                        + text_hidden
                    )
                    codec_input_embeds_list.append(pos_embed)

                # EOS token
                last_pos = num_codec_tokens - 1
                if last_pos < sample_trailing_len:
                    eos_text_hidden = sample_trailing_hidden[
                        :, last_pos : last_pos + 1, :
                    ]
                else:
                    eos_text_hidden = tts_pad_embed
                eos_input_embed = (
                    sample_all_layer_sum[:, last_pos : last_pos + 1, :]
                    + eos_text_hidden
                )
                codec_input_embeds_list.append(eos_input_embed)

                if codec_input_embeds_list:
                    sample_codec_inputs = torch.cat(codec_input_embeds_list, dim=1)
                    batch_codec_inputs.append(sample_codec_inputs)

            # Concatenate all samples
            if batch_codec_inputs:
                codec_input_embeds = torch.cat(batch_codec_inputs, dim=0).to(
                    self.talker.dtype
                )
                assert codec_input_embeds.shape == (
                    batch_size,
                    num_codec_tokens,
                    layer0_embeddings.shape[2],
                ), f"codec_input_embeds shape mismatch: {codec_input_embeds.shape}"
                full_inputs_embeds = torch.cat(
                    [talker_input_embed, codec_input_embeds], dim=1
                )
            else:
                full_inputs_embeds = talker_input_embed

            # Assert final input shape
            assert full_inputs_embeds.ndim == 3, (
                f"full_inputs_embeds should be 3D, got {full_inputs_embeds.shape}"
            )

            # --- Labels for Talker (Layer 0) ---
            # Labels: predict code[0] at prefix[-1], code[1] at shifted_input[0], ..., EOS at shifted_input[-1]
            # NOTE: prefix_len is PADDED max length
            prefix_len = talker_input_embed.shape[1]
            codec_eos_id = self.config.talker_config.codec_eos_token_id

            # Build per-sample labels to handle variable prefix lengths and padding gap
            labels_list = []
            for batch_idx in range(batch_size):
                sample_prefix_len = original_prefix_lens[batch_idx].item()

                # 1. Prefix labels: -100 for [0 ... L-2]
                # Length: sample_prefix_len - 1
                prefix_mask_len = int(sample_prefix_len - 1)
                prefix_mask = torch.full(
                    (1, prefix_mask_len), -100, dtype=torch.long, device=self.device
                )

                # 2. Label for code[0]: at [L-1]
                # code[0] is target_codes[batch, 0, 0]
                # But layer0_codes is [batch, num_tokens], so layer0_codes[batch, 0]
                label_code0 = layer0_codes[batch_idx : batch_idx + 1, 0:1]  # [1, 1]

                # 3. Padding gap labels: -100 for [L ... max_prefix-1]
                # Length: prefix_len - sample_prefix_len
                gap_len = int(prefix_len - sample_prefix_len)
                gap_mask = torch.full(
                    (1, gap_len), -100, dtype=torch.long, device=self.device
                )

                # 4. Labels for code[1:] + EOS: at [max_prefix ... end]
                # layer0_codes[1:] + EOS
                label_code_rest = layer0_codes[
                    batch_idx : batch_idx + 1, 1:
                ]  # [1, N-1]
                label_eos = torch.full(
                    (1, 1), codec_eos_id, dtype=torch.long, device=self.device
                )

                # Concatenate all parts
                # Structure: [PrefixMask, Code0, GapMask, CodeRest, EOS]
                # Total length: (L-1) + 1 + (max-L) + (N-1) + 1
                #             = L + max - L + N
                #             = max + N
                # This matches input length: max_prefix_len + num_codec_tokens

                sample_labels = torch.cat(
                    [prefix_mask, label_code0, gap_mask, label_code_rest, label_eos],
                    dim=1,
                )
                labels_list.append(sample_labels)

            labels = torch.cat(labels_list, dim=0)
            assert labels.shape == (batch_size, prefix_len + num_codec_tokens), (
                f"Labels shape mismatch: {labels.shape} vs expected ({batch_size}, {prefix_len + num_codec_tokens})"
            )

            # --- Attention Mask ---
            # Build proper attention mask that masks padded prefix positions (the GAP)
            seq_len = full_inputs_embeds.shape[1]
            attention_mask = torch.zeros(
                (batch_size, seq_len), dtype=torch.long, device=self.device
            )
            for batch_idx in range(batch_size):
                sample_prefix_len = original_prefix_lens[batch_idx].item()

                # Unmask prefix: [0 : sample_prefix_len]
                attention_mask[batch_idx, :sample_prefix_len] = 1

                # Unmask codec inputs: [max_prefix_len : end]
                # These are always valid as they contain the codec tokens
                attention_mask[batch_idx, prefix_len:] = 1

            assert attention_mask.shape == (batch_size, seq_len), (
                f"Attention mask shape mismatch: {attention_mask.shape}"
            )
            assert labels.shape[1] == seq_len, (
                f"Labels length {labels.shape[1]} != sequence length {seq_len}"
            )

            talker_outputs = self.talker(
                inputs_embeds=full_inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                trailing_text_hidden=trailing_text_hidden,
                tts_pad_embed=tts_pad_embed,
                talker_input_ids=None,
                audio_feature_lengths=None,
                image_grid_thw=None,
                video_grid_thw=None,
                video_second_per_grid=None,
                output_hidden_states=True,
            )
            talker_loss = talker_outputs.loss

            # ========================================
            # Part 2: MTP Training (Layers 1-15)
            # ========================================
            if not self.config.train_mtp:
                # Skip MTP training if flag is False
                mtp_avg_loss = torch.tensor(0.0, device=self.device)
            else:
                # Get Talker's last hidden state
                # NOTE: Talker returns hidden_states as (hidden_states_tuple, ...)
                # where hidden_states_tuple[0] is the standard tuple of layer outputs
                # hidden_states_tuple[0][-1] gives the last layer's output tensor
                if isinstance(talker_outputs.hidden_states, tuple):
                    talker_hidden = talker_outputs.hidden_states[0][-1]  # type: ignore[index]
                else:
                    talker_hidden = talker_outputs.hidden_states[-1]  # type: ignore[index]

                # Extract hidden states for codec token positions
                # CRITICAL: Need to extract per-sample using original_prefix_lens
                # AND jump over the padding gap!

                codec_hidden_list = []
                for batch_idx in range(batch_size):
                    sample_prefix_len = original_prefix_lens[batch_idx].item()

                    # 1. Hidden state at L-1 (for code[0])
                    # Shape: [1, 1, hidden]
                    h0 = talker_hidden[
                        batch_idx : batch_idx + 1,
                        sample_prefix_len - 1 : sample_prefix_len,
                        :,
                    ]

                    # 2. Hidden states for code[1:] (at max_prefix_len ... end-1)
                    # We need N-1 tokens.
                    # Input block starts at prefix_len.
                    # We want hidden states corresponding to inputs for code[1]...code[N-1]
                    # Inputs are at indices: prefix_len, prefix_len+1, ...
                    # So we take prefix_len : prefix_len + (num_codec_tokens - 1)
                    h_rest = talker_hidden[
                        batch_idx : batch_idx + 1,
                        prefix_len : prefix_len + num_codec_tokens - 1,
                        :,
                    ]

                    # Concatenate: [h0, h_rest] -> total N tokens
                    sample_codec_hidden = torch.cat([h0, h_rest], dim=1)
                    codec_hidden_list.append(sample_codec_hidden)

                codec_hidden = torch.cat(codec_hidden_list, dim=0)
                assert codec_hidden.shape == (
                    batch_size,
                    num_codec_tokens,
                    talker_hidden.shape[2],
                ), f"codec_hidden shape mismatch: {codec_hidden.shape}"

                mtp_total_loss: torch.Tensor = torch.tensor(0.0, device=self.device)
                code_predictor = self._get_unwrapped_code_predictor()

                hidden_dim = codec_hidden.shape[2]

                # Get layer 0 embeddings (already have layer0_embeddings from above)
                layer0_embed_for_mtp = layer0_embeddings

                # Reshape for batch processing: [batch * num_tokens, ...]
                hidden_flat = codec_hidden.reshape(-1, 1, hidden_dim)
                layer0_flat = layer0_embed_for_mtp.reshape(-1, 1, hidden_dim)

                # Train each MTP layer (15 layers total: layers 1-15)
                num_mtp_layers = 15
                for mtp_layer_idx in range(num_mtp_layers):
                    # Build input: [hidden, layer0, layer1, ..., layer_{mtp_layer_idx}]
                    embed_list = [hidden_flat, layer0_flat]

                    # Add previous layer embeddings (teacher forcing)
                    for prev_layer in range(mtp_layer_idx):
                        prev_codes = target_codes[:, prev_layer + 1, :].to(self.device)
                        # Assert shape
                        assert prev_codes.shape == (batch_size, num_codec_tokens), (
                            f"MTP layer {mtp_layer_idx}: prev_codes shape mismatch: {prev_codes.shape}"
                        )

                        prev_embed = predictor_embeds[prev_layer](prev_codes)
                        prev_embed_flat = prev_embed.reshape(-1, 1, hidden_dim)
                        embed_list.append(prev_embed_flat)

                    mtp_inputs = torch.cat(embed_list, dim=1).to(self.talker.dtype)

                    # Target: layer_{mtp_layer_idx + 1}
                    target_layer_codes = target_codes[:, mtp_layer_idx + 1, :].to(
                        self.device
                    )
                    # Assert target shape
                    assert target_layer_codes.shape == (batch_size, num_codec_tokens), (
                        f"MTP layer {mtp_layer_idx}: target_layer_codes shape mismatch: {target_layer_codes.shape}"
                    )

                    target_labels = target_layer_codes.reshape(-1)

                    # Forward through CodePredictor
                    mtp_outputs = code_predictor(
                        input_ids=None,
                        inputs_embeds=mtp_inputs,
                        generation_steps=mtp_layer_idx,
                        use_cache=False,
                    )

                    # Get logits for the last position
                    # MTP input seq length is (mtp_layer_idx + 2) [hidden, layer0, ... previous_layers]
                    # We predict the NEXT layer at the LAST position
                    mtp_logits = mtp_outputs.logits[:, -1, :]

                    # Assert logits shape
                    # vocab_size should match codebook size (usually 2048 or 4096)
                    assert mtp_logits.shape[0] == batch_size * num_codec_tokens, (
                        f"MTP layer {mtp_layer_idx}: logits batch size mismatch"
                    )

                    # Compute loss
                    mtp_layer_loss = F.cross_entropy(mtp_logits, target_labels)
                    mtp_total_loss += mtp_layer_loss

                # Average MTP loss across layers
                mtp_avg_loss = mtp_total_loss / num_mtp_layers

        # ==================== Phase 3: Combine Losses & Return ====================
        # Combined loss: Thinker + Talker + MTP
        mtp_weight = 2.0

        # Handle case where thinker_loss might be None (no labels provided)
        if thinker_loss is None:
            thinker_loss = torch.tensor(0.0, device=self.device)

        total_loss = thinker_loss + talker_loss + (mtp_weight * mtp_avg_loss)

        # Store individual losses for logging (accessible via model attributes)
        self._last_thinker_loss = thinker_loss.detach()
        self._last_talker_loss = talker_loss.detach()
        self._last_mtp_loss = mtp_avg_loss.detach()

        # Return output in HuggingFace format
        return CausalLMOutputWithPast(
            loss=total_loss,  # type: ignore
            logits=thinker_outputs.logits,
            past_key_values=thinker_outputs.past_key_values,
            hidden_states=thinker_outputs.hidden_states,
            attentions=thinker_outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        speaker_embedding: Optional[torch.Tensor] = None,
        speaker: str = "Ethan",
        use_audio_in_video: bool = False,
        return_audio: Optional[bool] = None,
        thinker_max_new_tokens: int = 1024,
        thinker_eos_token_id: int = 151645,
        talker_max_new_tokens: int = 4096,
        talker_do_sample: bool = True,
        talker_top_k: int = 50,
        talker_top_p: float = 1.0,
        talker_temperature: float = 0.9,
        talker_repetition_penalty: float = 1.05,
        **kwargs,
    ) -> Tuple[Any, Optional[torch.Tensor]]:
        """
        Generate text and optionally audio from multimodal input.

        This method supports voice cloning via speaker embeddings. When speaker_embedding
        is provided, it uses the continuous embedding for voice synthesis. Otherwise,
        it falls back to using discrete speaker IDs (like the original model).

        Args:
            input_ids: Input token IDs with multimodal placeholders [1, seq_len]
            speaker_embedding: Optional speaker embedding [1, 192] from ECAPA-TDNN for voice cloning.
                              If None, falls back to discrete speaker ID from `speaker` parameter.
            speaker: Speaker name for fallback when speaker_embedding is None. Default: "Ethan"
            use_audio_in_video: If True, use audio track from video inputs
            return_audio: If True, generate audio output. If None, auto-detect based on has_talker
            thinker_max_new_tokens: Max tokens for thinker (text) generation
            thinker_eos_token_id: EOS token ID for thinker
            talker_max_new_tokens: Max tokens for talker (audio codec) generation
            talker_do_sample: Whether to use sampling for talker
            talker_top_k: Top-k sampling parameter for talker
            talker_top_p: Top-p (nucleus) sampling parameter for talker
            talker_temperature: Temperature for talker sampling
            talker_repetition_penalty: Repetition penalty for talker
            **kwargs: Additional arguments (input_features, attention_mask, etc.)

        Returns:
            Tuple of (thinker_result, audio_waveform):
                - thinker_result: Text generation output from thinker
                - audio_waveform: Generated audio waveform tensor, or None if return_audio=False
        """
        # Validate talker availability
        if return_audio and not self.has_talker:
            raise ValueError(
                "Cannot use talker when talker module not initialized. "
                "Use `enable_talker` method or set enable_talker in config to enable talker."
            )
        if return_audio is None:
            return_audio = self.has_talker

        # Validate speaker_embedding shape if provided
        if speaker_embedding is not None:
            if speaker_embedding.ndim != 2 or speaker_embedding.shape[1] != 192:
                raise ValueError(
                    f"speaker_embedding must be [1, 192], got {speaker_embedding.shape}"
                )
            speaker_embedding = speaker_embedding.to(self.device)

        # ========================================
        # Parse kwargs into component-specific dicts
        # ========================================
        shared_kwargs = {"use_audio_in_video": use_audio_in_video}
        thinker_kwargs = {
            "max_new_tokens": thinker_max_new_tokens,
            "eos_token_id": thinker_eos_token_id,
        }

        talker_kwargs = {}
        token2wav_kwargs = {}

        if return_audio:
            # Validate batch size (original limitation)
            if input_ids is not None and input_ids.shape[0] != 1:
                raise NotImplementedError(
                    "Qwen3-Omni currently does not support batched inference with audio output"
                )

            # Build suppressed tokens list (special tokens that shouldn't be predicted)
            talker_suppressed_tokens = [
                i
                for i in range(
                    self.config.talker_config.text_config.vocab_size - 1024,
                    self.config.talker_config.text_config.vocab_size,
                )
                if i != self.config.talker_config.codec_eos_token_id
            ]

            talker_kwargs = {
                "max_new_tokens": talker_max_new_tokens,
                "do_sample": talker_do_sample,
                "top_k": talker_top_k,
                "top_p": talker_top_p,
                "temperature": talker_temperature,
                "eos_token_id": self.config.talker_config.codec_eos_token_id,
                "repetition_penalty": talker_repetition_penalty,
                "suppress_tokens": talker_suppressed_tokens,
                "output_hidden_states": True,
                "return_dict_in_generate": True,
            }

        # Parse prefixed kwargs
        for key, value in kwargs.items():
            if key.startswith("thinker_"):
                thinker_kwargs[key[len("thinker_") :]] = value
            elif key.startswith("talker_"):
                talker_kwargs[key[len("talker_") :]] = value
            elif key.startswith("token2wav_"):
                token2wav_kwargs[key[len("token2wav_") :]] = value
            elif key == "feature_attention_mask":
                thinker_kwargs[key] = value
                talker_kwargs["audio_feature_lengths"] = torch.sum(value, dim=1)
            elif key in ("input_features", "attention_mask"):
                thinker_kwargs[key] = value
            else:
                shared_kwargs[key] = value

        # Merge shared kwargs
        for key, value in shared_kwargs.items():
            if key not in thinker_kwargs:
                thinker_kwargs[key] = value
            if key not in talker_kwargs and key in [
                "image_grid_thw",
                "video_grid_thw",
                "video_second_per_grid",
            ]:
                talker_kwargs[key] = value
            if key not in token2wav_kwargs:
                token2wav_kwargs[key] = value

        # ========================================
        # Step 1: Thinker Generation (Text)
        # ========================================
        generate_audio = return_audio and self.has_talker
        if generate_audio:
            thinker_kwargs["output_hidden_states"] = True
            thinker_kwargs["return_dict_in_generate"] = True

        thinker_result = self.thinker.generate(input_ids=input_ids, **thinker_kwargs)

        if not generate_audio:
            return thinker_result, None

        # ========================================
        # Step 2: Prepare Talker Input
        # ========================================
        # Extract hidden states from thinker generation
        # thinker_result.hidden_states is a tuple of (per-token hidden states)
        # Each element is a tuple of layer outputs, we need layer 0 and accept_hidden_layer
        thinker_embed = torch.cat(
            [hidden_states[0] for hidden_states in thinker_result.hidden_states], dim=1
        ).to(self.talker.device)  # [1, total_seq, hidden]

        accept_layer = self.config.talker_config.accept_hidden_layer
        thinker_hidden = torch.cat(
            [
                hidden_states[accept_layer]
                for hidden_states in thinker_result.hidden_states
            ],
            dim=1,
        ).to(self.talker.device)  # [1, total_seq, hidden]

        # Find im_start positions in the full sequence (input + generated)
        im_start_indexes = torch.cat(
            (
                torch.nonzero(input_ids[0] == self.config.im_start_token_id).squeeze(),
                torch.tensor(
                    [thinker_result.sequences.shape[-1]],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                ),
            ),
            dim=-1,
        ).to(self.talker.device)

        # Build multimodal mask for the full sequence
        multimodal_mask = (
            (thinker_result.sequences == self.config.thinker_config.audio_token_id)
            | (thinker_result.sequences == self.config.thinker_config.image_token_id)
            | (thinker_result.sequences == self.config.thinker_config.video_token_id)
        ).to(self.talker.device)

        # Get TTS special token embeddings
        talker_special_tokens = torch.tensor(
            [
                [
                    self.config.tts_bos_token_id,
                    self.config.tts_eos_token_id,
                    self.config.tts_pad_token_id,
                ]
            ],
            device=self.thinker.device,
            dtype=input_ids.dtype,
        )

        # Handle LoRA-wrapped thinker embeddings
        thinker_embeddings = self.thinker.get_input_embeddings()
        if hasattr(thinker_embeddings, "base_layer"):
            thinker_embeddings = thinker_embeddings.base_layer

        tts_bos_embed, tts_eos_embed, tts_pad_embed = (
            self.talker.text_projection(thinker_embeddings(talker_special_tokens))
            .to(self.talker.device)
            .chunk(3, dim=1)
        )

        # ========================================
        # Step 3: Build Talker Prefix
        # ========================================
        talker_input_embeds = []
        talker_input_ids = []
        trailing_text_hidden = None

        for i in range(len(im_start_indexes) - 1):
            im_start_index = im_start_indexes[i]
            segment_end_index = im_start_indexes[i + 1]
            role_token = input_ids[0][im_start_index + 1]

            # Skip system prompts
            if role_token == self.config.system_token_id:
                continue

            # User turn: use text projection + hidden projection for multimodal
            elif role_token == self.config.user_token_id:
                talker_user_part = self._get_talker_user_parts(
                    im_start_index,
                    segment_end_index,
                    multimodal_mask,
                    thinker_hidden,
                    thinker_embed,
                )
                talker_input_embeds.append(talker_user_part)
                talker_input_ids.append(
                    thinker_result.sequences[:, im_start_index:segment_end_index]
                )

            # Current assistant turn (last one): build with speaker embedding
            elif (
                role_token == self.config.assistant_token_id
                and i == len(im_start_indexes) - 2
            ):
                if speaker_embedding is not None:
                    # Use speaker embedding for voice cloning
                    assistant_embeds, assistant_ids, trailing_text_hidden = (
                        self._get_talker_assistant_parts(
                            im_start_index,
                            segment_end_index,
                            speaker_embedding,
                            thinker_embed,
                            tts_pad_embed,
                            tts_bos_embed,
                            tts_eos_embed,
                        )
                    )
                else:
                    # Fallback to discrete speaker ID (original behavior)
                    speaker_id = self.config.talker_config.speaker_id.get(
                        speaker.lower()
                    )
                    if speaker_id is None:
                        raise NotImplementedError(f"Speaker {speaker} not implemented")

                    # Call parent's _get_talker_assistant_parts with speaker_id
                    assistant_embeds, assistant_ids, trailing_text_hidden = (
                        Qwen3OmniMoeForConditionalGeneration._get_talker_assistant_parts(
                            self,
                            im_start_index,
                            segment_end_index,
                            speaker_id,
                            thinker_embed,
                            tts_pad_embed,
                            tts_bos_embed,
                            tts_eos_embed,
                        )
                    )

                talker_input_embeds.append(assistant_embeds)
                talker_input_ids.append(assistant_ids)

            # Historical assistant turns: skip (same as original)
            elif (
                role_token == self.config.assistant_token_id
                and i != len(im_start_indexes) - 2
            ):
                continue

            else:
                raise AssertionError(
                    "Expect role id after <|im_start|> (assistant, user, system)"
                )

        if trailing_text_hidden is None:
            raise RuntimeError(
                "Failed to build trailing_text_hidden for talker generation."
            )

        talker_input_embed = torch.cat(
            [embed.to(self.talker.device) for embed in talker_input_embeds], dim=1
        )
        talker_input_id = torch.cat(
            [ids.to(self.talker.device) for ids in talker_input_ids], dim=1
        )

        # ========================================
        # Step 4: Talker Generation (Codec Tokens)
        # ========================================
        talker_result = self.talker.generate(
            inputs_embeds=talker_input_embed,
            trailing_text_hidden=trailing_text_hidden,
            tts_pad_embed=tts_pad_embed,
            talker_input_ids=talker_input_id,
            **talker_kwargs,
        )

        # ========================================
        # Step 5: Convert Codec Tokens to Waveform
        # ========================================
        # Extract codec tokens from talker hidden states
        # talker_result.hidden_states is (hidden_states_tuple, residual_codes) per token
        talker_codes = (
            torch.stack(
                [hid[-1] for hid in talker_result.hidden_states if hid[-1] is not None],
                dim=1,
            )
            .transpose(1, 2)
            .to(self.code2wav.device)
        )

        # Decode to waveform using code2wav
        talker_wavs = self.code2wav.chunked_decode(
            talker_codes, chunk_size=300, left_context_size=25
        )

        return thinker_result, talker_wavs.float()
