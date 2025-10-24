"""
Text Encoder Module
Implements text feature extraction based on CLIP, supporting encoding of positive and negative prompts
"""

import torch
import torch.nn as nn
import logging
from typing import List, Union
import open_clip


class TextEncoder(nn.Module):
    """
    CLIP-based text encoder for encoding text prompts
    """
    def __init__(self, device, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        super(TextEncoder, self).__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model_name = model_name
        self.pretrained = pretrained

        try:
            # Load pre-trained CLIP model, first create on CPU
            self.clip_model, _, _ = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device='cpu'  # Create on CPU first, then manually move to target device
            )
            self.tokenizer = open_clip.get_tokenizer(model_name)

            # Manually move model to target device
            self.clip_model = self.clip_model.to(self.device)
            logging.info(f"Successfully loaded CLIP model: {model_name} with {pretrained} on {self.device}")
        except Exception as e:
            logging.error(f"Failed to load CLIP model: {e}")
            # Fallback to base model
            self.clip_model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-32",
                pretrained="openai",
                device='cpu'  # Create on CPU first
            )
            self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
            # Manually move model to target device
            self.clip_model = self.clip_model.to(self.device)
            logging.warning(f"Fallback to ViT-B-32 with openai weights on {self.device}")
    
    def forward(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode text to 512-dimensional feature vectors
        Args:
            texts: String or list of strings
        Returns:
            text_features: Text features of shape (batch_size, 512)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            tokens = self.tokenizer(texts).to(self.device)
            with torch.no_grad():  # No gradient needed for text encoding
                text_features = self.clip_model.encode_text(tokens)
            return text_features.float()
        except Exception as e:
            logging.error(f"Text encoding failed: {e}")
            # Return zero vector as fallback
            return torch.zeros(len(texts), 512, device=self.device)


class PosNegTextEncoder(nn.Module):
    """
    Text encoder for processing positive and negative prompts
    Supports batch processing of varying numbers of negative prompts
    """
    def __init__(self, device, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        super(PosNegTextEncoder, self).__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.text_encoder = TextEncoder(device=self.device, model_name=model_name, pretrained=pretrained)

    def to(self, device):
        """Override to() method to ensure internal text_encoder also moves to correct device"""
        self.device = torch.device(device) if isinstance(device, str) else device
        result = super().to(device)
        if hasattr(self.text_encoder, 'clip_model'):
            self.text_encoder.clip_model = self.text_encoder.clip_model.to(device)
            self.text_encoder.device = self.device
        return result
        
    def encode_positive(self, texts: List[str]) -> torch.Tensor:
        """
        Encode positive prompts
        Args:
            texts: List of positive prompts, length equals batch_size
        Returns:
            pos_embeddings: Positive prompt embeddings of shape (batch_size, 512)
        """
        return self.text_encoder(texts)
    
    def encode_negative(self, texts: List[List[str]]) -> torch.Tensor:
        """
        Encode negative prompts
        Args:
            texts: Nested list, each sample contains multiple negative prompts
                  Format: [[neg1_1, neg1_2, ...], [neg2_1, neg2_2, ...], ...]
        Returns:
            neg_embeddings: Negative prompt embeddings of shape (batch_size, num_neg_prompts, 512)
        """
        if not texts or not texts[0]:
            # If no negative prompts, return zero vector
            return torch.zeros(len(texts), 1, 512, device=self.device)

        B = len(texts)  # Batch size
        # logging.info(f"Encoding negative prompts - batch size: {B}, input texts structure: {[len(x) for x in texts]}")

        # Find maximum number of negative prompts for uniform padding
        max_neg_prompts = max(len(sample_neg_prompts) for sample_neg_prompts in texts)
        if max_neg_prompts == 0:
            return torch.zeros(B, 1, 512, device=self.device)

        # Normalize number of negative prompts for all samples
        normalized_texts = []
        for i, sample_neg_prompts in enumerate(texts):
            if len(sample_neg_prompts) < max_neg_prompts:
                # Pad with empty strings to uniform length
                padded_prompts = sample_neg_prompts + [""] * (max_neg_prompts - len(sample_neg_prompts))
                normalized_texts.append(padded_prompts)
                logging.debug(f"Sample {i} padded from {len(sample_neg_prompts)} to {max_neg_prompts} negative prompts")
            elif len(sample_neg_prompts) > max_neg_prompts:
                # Truncate to uniform length
                truncated_prompts = sample_neg_prompts[:max_neg_prompts]
                normalized_texts.append(truncated_prompts)
                logging.warning(f"Sample {i} truncated from {len(sample_neg_prompts)} to {max_neg_prompts} negative prompts")
            else:
                normalized_texts.append(sample_neg_prompts)

        # Flatten all negative prompts while maintaining scene-negative sample relationships
        all_neg_prompts = []
        for sample_neg_prompts in normalized_texts:
            all_neg_prompts.extend(sample_neg_prompts)

        # Batch encode all negative prompts
        embeddings = self.text_encoder(all_neg_prompts)   # (B * max_neg_prompts, 512)

        # Reshape to (B, max_neg_prompts, 512) - preserve semantic relationships
        # logging.info(f"Encoded negative prompts: {embeddings.shape}")
        embeddings = embeddings.reshape(B, max_neg_prompts, embeddings.shape[-1])
        # logging.info(f"Reshaped negative prompts: {embeddings.shape}")
        return embeddings
    
    def forward(self, texts: Union[List[str], List[List[str]]], type: str) -> torch.Tensor:
        """
        Unified forward interface
        Args:
            texts: Text input
            type: "pos" or "neg"
        Returns:
            embeddings: Text embeddings
        """
        if type == "pos":
            return self.encode_positive(texts)
        elif type == "neg":
            return self.encode_negative(texts)
        else:
            raise ValueError(f"Unknown type: {type}. Must be 'pos' or 'neg'")


class TextConditionProcessor(nn.Module):
    """
    Text condition processor for handling text conditions in diffusion models
    """
    def __init__(self, text_dim=512, context_dim=512, dropout=0.1, use_negative_prompts=True):
        super(TextConditionProcessor, self).__init__()
        self.text_dim = text_dim
        self.context_dim = context_dim
        self.use_negative_prompts = use_negative_prompts

        # Negative prompt prediction network
        # Input: scene_features - pos_text_features (both 512-dimensional)
        # Output: CLIP embedding dimension, consistent with text_cond
        if self.use_negative_prompts:
            self.negative_net = nn.Sequential(
                nn.Linear(text_dim, text_dim),  # 512 -> 512 (input is 512-dim difference)
                nn.LayerNorm(text_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(text_dim, text_dim)   # Output CLIP embedding dimension
            )
        else:
            self.negative_net = None
        
    def forward(self, pos_text_features, neg_text_features=None, scene_features=None):
        """
        Process text conditions and predict distractors

        Design concept:
        - negative_net receives the "scene-target" difference as input
        - This difference represents "semantic information in the scene except for the target object"
        - negative_net learns to predict significant distractors from this residual information

        Args:
            pos_text_features: (B, text_dim) Positive text features (target object description)
            neg_text_features: (B, num_neg, text_dim) Real distractor features, only for loss computation
            scene_features: (B, text_dim) Global scene features
        Returns:
            pos_text_features: (B, text_dim) Original positive text features (returned directly, no projection)
            neg_pred: (B, text_dim) Predicted distractor CLIP embeddings
        """
        neg_pred = None
        if self.use_negative_prompts and self.negative_net is not None and \
           neg_text_features is not None and scene_features is not None:
            # Predict negative prompts based on scene-target difference (pure negative_net prediction)
            # Let the network learn to identify distractors from the scene, without mixing real negative prompt information
            neg_pred = self.negative_net(scene_features - pos_text_features)  # (B, text_dim)

        return pos_text_features, neg_pred