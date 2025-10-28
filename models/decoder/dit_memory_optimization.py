"""
Memory optimization utilities for DiT model.

This module provides memory optimization features including:
- Gradient checkpointing support
- Efficient attention computation
- Memory usage monitoring
- Scalable batch processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import psutil
import logging
from typing import Optional, Tuple, Dict, Any
from contextlib import contextmanager
import gc
import math

# Try to import PyTorch 2.x SDPA (Scaled Dot-Product Attention)
# This is the recommended approach for PyTorch 2.0+
_SDPA_AVAILABLE = False
try:
    # Check if scaled_dot_product_attention is available
    if hasattr(F, 'scaled_dot_product_attention'):
        # Test if it works (some PyTorch builds may have it but not working)
        _test_q = torch.randn(1, 1, 1, 8)
        _test_k = torch.randn(1, 1, 1, 8)
        _test_v = torch.randn(1, 1, 1, 8)
        _ = F.scaled_dot_product_attention(_test_q, _test_k, _test_v, dropout_p=0.0)
        _SDPA_AVAILABLE = True
        logging.info("PyTorch 2.x SDPA (scaled_dot_product_attention) is available and will be used")
    else:
        logging.warning("PyTorch 2.x SDPA not available. Using fallback attention implementation.")
except Exception as e:
    logging.warning(f"PyTorch 2.x SDPA test failed: {e}. Using fallback attention implementation.")
    _SDPA_AVAILABLE = False


class MemoryMonitor:
    """
    Memory usage monitoring and optimization hints.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.peak_memory = 0
        self.initial_memory = 0
        
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_cached = torch.cuda.memory_reserved() / 1024**3   # GB
            gpu_max = torch.cuda.max_memory_allocated() / 1024**3  # GB
        else:
            gpu_memory = gpu_cached = gpu_max = 0.0
            
        cpu_memory = psutil.virtual_memory().used / 1024**3  # GB
        cpu_percent = psutil.virtual_memory().percent
        
        return {
            'gpu_allocated_gb': gpu_memory,
            'gpu_cached_gb': gpu_cached,
            'gpu_max_allocated_gb': gpu_max,
            'cpu_used_gb': cpu_memory,
            'cpu_percent': cpu_percent
        }
    
    def log_memory_usage(self, stage: str = ""):
        """Log current memory usage."""
        info = self.get_memory_info()
        self.logger.info(
            f"Memory usage {stage}: "
            f"GPU: {info['gpu_allocated_gb']:.2f}GB allocated, "
            f"{info['gpu_cached_gb']:.2f}GB cached, "
            f"CPU: {info['cpu_used_gb']:.2f}GB ({info['cpu_percent']:.1f}%)"
        )
    
    def check_memory_pressure(self) -> Dict[str, Any]:
        """Check if system is under memory pressure and provide optimization hints."""
        info = self.get_memory_info()
        hints = []
        
        # GPU memory pressure
        if torch.cuda.is_available():
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_usage_percent = (info['gpu_allocated_gb'] / gpu_total) * 100
            
            if gpu_usage_percent > 90:
                hints.append("Critical GPU memory usage (>90%). Consider enabling gradient checkpointing.")
            elif gpu_usage_percent > 75:
                hints.append("High GPU memory usage (>75%). Consider reducing batch size or enabling optimizations.")
        
        # CPU memory pressure
        if info['cpu_percent'] > 90:
            hints.append("Critical CPU memory usage (>90%). Consider reducing data loading workers.")
        elif info['cpu_percent'] > 75:
            hints.append("High CPU memory usage (>75%). Monitor for memory leaks.")
        
        return {
            'memory_info': info,
            'optimization_hints': hints,
            'under_pressure': len(hints) > 0
        }
    
    @contextmanager
    def monitor_peak_memory(self, operation_name: str = "operation"):
        """Context manager to monitor peak memory usage during an operation."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        else:
            initial_memory = 0
            
        try:
            yield
        finally:
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated()
                memory_used = (peak_memory - initial_memory) / 1024**3
                self.logger.info(f"{operation_name} peak memory usage: {memory_used:.2f}GB")


class EfficientAttention(nn.Module):
    """
    Memory-efficient attention computation for large sequence lengths.
    
    Implements several optimization strategies (in priority order):
    1. PyTorch 2.x SDPA (scaled_dot_product_attention) - preferred, auto-optimized
    2. Flash Attention 2 (if explicitly requested and available)
    3. Chunked attention computation (for very long sequences)
    4. Standard attention (fallback)
    
    Features:
    - Automatic backend selection with PyTorch 2.x SDPA
    - Attention probability dropout for regularization
    - Mask support for handling padding
    """
    
    def __init__(self, d_model: int, num_heads: int, d_head: int, 
                 dropout: float = 0.1, chunk_size: int = 512,
                 use_flash_attention: bool = False, 
                 attention_dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.inner_dim = num_heads * d_head
        self.scale = d_head ** -0.5
        self.chunk_size = chunk_size
        self.use_flash_attention = use_flash_attention
        self.attention_dropout = attention_dropout
        
        # Check if SDPA is available
        self.use_sdpa = _SDPA_AVAILABLE
        
        self.to_q = nn.Linear(d_model, self.inner_dim, bias=False)
        self.to_k = nn.Linear(d_model, self.inner_dim, bias=False)
        self.to_v = nn.Linear(d_model, self.inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, d_model),
            nn.Dropout(dropout)
        )
        
        # Attention probability dropout (applied after softmax)
        # Only used for non-SDPA paths (SDPA has built-in dropout support)
        self.attn_dropout = nn.Dropout(attention_dropout) if attention_dropout > 0.0 else None
        
        # Try to import flash attention if explicitly requested (fallback option)
        self.flash_attn_func = None
        if use_flash_attention and not self.use_sdpa:
            try:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
                logging.info("Flash Attention 2 loaded as fallback (SDPA not available)")
            except ImportError:
                logging.warning("Flash attention not available, using standard attention")
    
    def forward(self, query: torch.Tensor, key: Optional[torch.Tensor] = None, 
                value: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None,
                q_scale: Optional[torch.Tensor] = None, q_shift: Optional[torch.Tensor] = None,
                k_scale: Optional[torch.Tensor] = None, k_shift: Optional[torch.Tensor] = None,
                v_scale: Optional[torch.Tensor] = None, v_shift: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Memory-efficient attention forward pass.
        
        Args:
            query: (B, seq_len_q, d_model)
            key: (B, seq_len_k, d_model) or None
            value: (B, seq_len_v, d_model) or None
            mask: Optional attention mask (1=valid, 0=padding)
                  Shape: (B, seq_len_k) or (B, seq_len_q, seq_len_k)
            q_scale/q_shift/k_scale/k_shift/v_scale/v_shift: Optional modulation tensors applied
                element-wise to the projected queries, keys, and values. Shapes should be broadcastable
                to (B, seq_len, num_heads * d_head) for the corresponding tensor.
            
        Returns:
            output: (B, seq_len_q, d_model)
        """
        if key is None:
            key = query
        if value is None:
            value = key
            
        B, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        modulation = {
            "q_scale": q_scale,
            "q_shift": q_shift,
            "k_scale": k_scale,
            "k_shift": k_shift,
            "v_scale": v_scale,
            "v_shift": v_shift,
        }
        
        # Priority 1: Use PyTorch 2.x SDPA (recommended)
        if self.use_sdpa:
            return self._sdpa_attention_forward(query, key, value, mask, **modulation)
        
        # Priority 2: Use flash attention if available and sequences are long enough
        if (self.flash_attn_func is not None and 
            seq_len_q > 128 and seq_len_k > 128 and 
            mask is None):  # Flash attention doesn't support custom masks easily
            return self._flash_attention_forward(query, key, value, **modulation)
        
        # Priority 3: Use chunked attention for very long sequences
        if seq_len_q > self.chunk_size or seq_len_k > self.chunk_size:
            return self._chunked_attention_forward(query, key, value, mask, **modulation)
        
        # Fallback: Standard attention for smaller sequences
        return self._standard_attention_forward(query, key, value, mask, **modulation)
    
    def _sdpa_attention_forward(self, query: torch.Tensor, key: torch.Tensor, 
                               value: torch.Tensor, mask: Optional[torch.Tensor] = None,
                               q_scale: Optional[torch.Tensor] = None, q_shift: Optional[torch.Tensor] = None,
                               k_scale: Optional[torch.Tensor] = None, k_shift: Optional[torch.Tensor] = None,
                               v_scale: Optional[torch.Tensor] = None, v_shift: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        PyTorch 2.x SDPA (Scaled Dot-Product Attention) implementation with optional Q/K/V modulation.
        
        This is the preferred method when available, as it automatically selects
        the best backend (FlashAttention 2, Memory-efficient, or Math) and provides
        native support for dropout and masks.
        """
        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)
        
        q = self._apply_modulation(q, q_scale, q_shift, "q")
        k = self._apply_modulation(k, k_scale, k_shift, "k")
        v = self._apply_modulation(v, v_scale, v_shift, "v")
        
        # Reshape for multi-head attention: (B, seq_len, num_heads, d_head)
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.d_head)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.d_head)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.d_head)
        
        # Convert mask format if provided
        # SDPA expects: (B, num_heads, seq_len_q, seq_len_k) with True=masked, False=valid
        # Our mask: (B, seq_len_k) or (B, seq_len_q, seq_len_k) with 1=valid, 0=padding
        attn_mask = None
        if mask is not None:
            if mask.dim() == 2:
                # (B, seq_len_k) -> (B, 1, 1, seq_len_k)
                attn_mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                # (B, seq_len_q, seq_len_k) -> (B, 1, seq_len_q, seq_len_k)
                attn_mask = mask.unsqueeze(1)
            
            # Convert: 1=valid, 0=padding -> False=valid, True=masked
            attn_mask = (attn_mask == 0)
        
        # Use SDPA with native dropout support
        dropout_p = self.attention_dropout if self.training else 0.0
        
        # PyTorch SDPA expects (B, num_heads, seq_len, d_head) but we have (B, seq_len, num_heads, d_head)
        # Transpose: (B, seq_len, num_heads, d_head) -> (B, num_heads, seq_len, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Call SDPA
        # Note: SDPA automatically calculates scale as 1/sqrt(d_head)
        # Our self.scale = d_head ** -0.5 is the same, so we don't need to pass it
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p
        )
        
        # Transpose back: (B, num_heads, seq_len, d_head) -> (B, seq_len, num_heads, d_head)
        out = out.transpose(1, 2)
        
        # Reshape back to (B, seq_len, d_model)
        out = out.contiguous().view(out.shape[0], out.shape[1], self.inner_dim)
        
        return self.to_out(out)
    
    def _flash_attention_forward(self, query: torch.Tensor, key: torch.Tensor, 
                                value: torch.Tensor,
                                q_scale: Optional[torch.Tensor] = None, q_shift: Optional[torch.Tensor] = None,
                                k_scale: Optional[torch.Tensor] = None, k_shift: Optional[torch.Tensor] = None,
                                v_scale: Optional[torch.Tensor] = None, v_shift: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Flash attention implementation with dropout support and optional Q/K/V modulation."""
        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)
        
        q = self._apply_modulation(q, q_scale, q_shift, "q")
        k = self._apply_modulation(k, k_scale, k_shift, "k")
        v = self._apply_modulation(v, v_scale, v_shift, "v")
        
        # Reshape for flash attention: (B, seq_len, num_heads, d_head)
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.d_head)
        k = k.view(k.shape[0], k.shape[1], self.num_heads, self.d_head)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.d_head)
        
        # Flash attention expects (B, seq_len, num_heads, d_head)
        # Pass attention_dropout to flash attention (it handles it internally)
        dropout_p = self.attention_dropout if self.training else 0.0
        out = self.flash_attn_func(q, k, v, dropout_p=dropout_p, softmax_scale=self.scale)
        
        # Reshape back to (B, seq_len, d_model)
        out = out.view(out.shape[0], out.shape[1], self.inner_dim)
        
        return self.to_out(out)
    
    def _chunked_attention_forward(self, query: torch.Tensor, key: torch.Tensor, 
                                  value: torch.Tensor, mask: Optional[torch.Tensor] = None,
                                  q_scale: Optional[torch.Tensor] = None, q_shift: Optional[torch.Tensor] = None,
                                  k_scale: Optional[torch.Tensor] = None, k_shift: Optional[torch.Tensor] = None,
                                  v_scale: Optional[torch.Tensor] = None, v_shift: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Chunked attention for memory efficiency with long sequences, dropout, and optional Q/K/V modulation."""
        B, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)
        
        q = self._apply_modulation(q, q_scale, q_shift, "q")
        k = self._apply_modulation(k, k_scale, k_shift, "k")
        v = self._apply_modulation(v, v_scale, v_shift, "v")
        
        # Reshape for multi-head attention
        q = q.view(B, seq_len_q, self.num_heads, self.d_head).transpose(1, 2)  # (B, num_heads, seq_len_q, d_head)
        k = k.view(B, seq_len_k, self.num_heads, self.d_head).transpose(1, 2)  # (B, num_heads, seq_len_k, d_head)
        v = v.view(B, seq_len_k, self.num_heads, self.d_head).transpose(1, 2)  # (B, num_heads, seq_len_k, d_head)
        
        # Process in chunks to save memory
        output_chunks = []
        
        for i in range(0, seq_len_q, self.chunk_size):
            end_i = min(i + self.chunk_size, seq_len_q)
            q_chunk = q[:, :, i:end_i, :]  # (B, num_heads, chunk_size, d_head)
            
            # Compute attention scores for this chunk
            scores = torch.einsum('bhid,bhjd->bhij', q_chunk, k) * self.scale
            
            if mask is not None:
                # Handle different mask shapes
                # Expected: mask should be broadcastable to (B, num_heads, chunk_size, seq_len_k)
                if mask.dim() == 2:
                    # (B, seq_len_k) -> (B, 1, 1, seq_len_k)
                    mask_chunk = mask.unsqueeze(1).unsqueeze(1)
                elif mask.dim() == 3:
                    # (B, seq_len_q, seq_len_k) -> (B, 1, chunk_size, seq_len_k)
                    mask_chunk = mask[:, i:end_i, :].unsqueeze(1)
                else:
                    mask_chunk = mask
                scores = scores.masked_fill(mask_chunk == 0, -float('inf'))
            
            attn = torch.softmax(scores, dim=-1)
            
            # Apply attention dropout (only during training)
            if self.attn_dropout is not None:
                attn = self.attn_dropout(attn)
            
            # Apply attention to values
            out_chunk = torch.einsum('bhij,bhjd->bhid', attn, v)
            output_chunks.append(out_chunk)
        
        # Concatenate chunks
        out = torch.cat(output_chunks, dim=2)  # (B, num_heads, seq_len_q, d_head)
        out = out.transpose(1, 2).contiguous().view(B, seq_len_q, self.inner_dim)
        
        return self.to_out(out)
    
    def _standard_attention_forward(self, query: torch.Tensor, key: torch.Tensor, 
                                   value: torch.Tensor, mask: Optional[torch.Tensor] = None,
                                   q_scale: Optional[torch.Tensor] = None, q_shift: Optional[torch.Tensor] = None,
                                   k_scale: Optional[torch.Tensor] = None, k_shift: Optional[torch.Tensor] = None,
                                   v_scale: Optional[torch.Tensor] = None, v_shift: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Standard attention implementation with dropout support and optional Q/K/V modulation."""
        from einops import rearrange
        
        B, seq_len_q, _ = query.shape
        
        q = self.to_q(query)
        k = self.to_k(key)
        v = self.to_v(value)
        
        q = self._apply_modulation(q, q_scale, q_shift, "q")
        k = self._apply_modulation(k, k_scale, k_shift, "k")
        v = self._apply_modulation(v, v_scale, v_shift, "v")
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Compute attention scores
        scores = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        
        if mask is not None:
            # Handle different mask shapes
            # Expected: mask should be broadcastable to (B, num_heads, seq_len_q, seq_len_k)
            if mask.dim() == 2:
                # (B, seq_len_k) -> (B, 1, 1, seq_len_k)
                mask = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                # (B, seq_len_q, seq_len_k) -> (B, 1, seq_len_q, seq_len_k)
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -float('inf'))
        
        attn = torch.softmax(scores, dim=-1)
        
        # Apply attention dropout (only during training)
        if self.attn_dropout is not None:
            attn = self.attn_dropout(attn)
        
        # Apply attention to values
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)
    
    def _prepare_modulation(self, modulation: torch.Tensor, reference: torch.Tensor, label: str) -> torch.Tensor:
        target_shape = reference.shape
        if modulation.dim() == 1:
            modulation = modulation.view(1, 1, -1)
        elif modulation.dim() == 2:
            if modulation.shape[0] in (1, target_shape[0]) and modulation.shape[1] == target_shape[2]:
                modulation = modulation.unsqueeze(1)
            elif modulation.shape[0] == target_shape[1] and modulation.shape[1] == target_shape[2]:
                modulation = modulation.unsqueeze(0)
            elif modulation.shape[0] in (1, target_shape[0]) and modulation.shape[1] == 1:
                modulation = modulation.unsqueeze(-1)
            else:
                modulation = modulation.unsqueeze(0).unsqueeze(0)
        elif modulation.dim() != 3:
            raise RuntimeError(
                f"Unsupported modulation rank for {label}: {modulation.dim()}D"
            )
        try:
            modulation = torch.broadcast_to(modulation, target_shape)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Unable to broadcast modulation {label} with shape {tuple(modulation.shape)} "
                f"to reference {tuple(target_shape)}"
            ) from exc
        return modulation
    
    def _apply_modulation(self, tensor: torch.Tensor,
                          scale: Optional[torch.Tensor],
                          shift: Optional[torch.Tensor],
                          name: str) -> torch.Tensor:
        if scale is not None:
            scale = self._prepare_modulation(scale.to(device=tensor.device, dtype=tensor.dtype), tensor, f"{name}_scale")
            tensor = tensor * (1 + scale)
        if shift is not None:
            shift = self._prepare_modulation(shift.to(device=tensor.device, dtype=tensor.dtype), tensor, f"{name}_shift")
            tensor = tensor + shift
        return tensor
    

class GradientCheckpointedDiTBlock(nn.Module):
    """
    DiT block with gradient checkpointing support for memory optimization.
    """

    def __init__(self, dit_block: nn.Module, use_checkpointing: bool = True):
        super().__init__()
        self.dit_block = dit_block
        self.use_checkpointing = use_checkpointing

    def forward(
        self,
        x: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
        scene_context: Optional[torch.Tensor] = None,
        text_context: Optional[torch.Tensor] = None,
        scene_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional gradient checkpointing.
        """
        if self.use_checkpointing and self.training:
            return checkpoint.checkpoint(
                self._forward_impl,
                x,
                time_emb,
                scene_context,
                text_context,
                scene_mask,
                text_mask,
                use_reentrant=False,
            )
        return self._forward_impl(
            x,
            time_emb,
            scene_context,
            text_context,
            scene_mask,
            text_mask,
        )

    def _forward_impl(
        self,
        x: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
        scene_context: Optional[torch.Tensor] = None,
        text_context: Optional[torch.Tensor] = None,
        scene_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Actual forward implementation."""
        return self.dit_block(
            x,
            time_emb,
            scene_context,
            text_context,
            scene_mask,
            text_mask=text_mask,
        )


class BatchProcessor:
    """
    Scalable batch processing for different grasp counts with memory optimization.
    """
    
    def __init__(
        self,
        max_batch_size: int = 32,
        max_sequence_length: int = 100,
        logger: Optional[logging.Logger] = None,
        default_d_model: int = 512,
        default_num_layers: int = 12,
    ):
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        self.logger = logger or logging.getLogger(__name__)
        self.memory_monitor = MemoryMonitor(logger)
        self.default_d_model = default_d_model
        self.default_num_layers = default_num_layers
    
    def estimate_memory_usage(self, batch_size: int, sequence_length: int, 
                            d_model: int, num_layers: int) -> float:
        """
        Estimate memory usage for given batch configuration.
        
        Returns:
            Estimated memory usage in GB
        """
        # Rough estimation based on transformer memory patterns
        # This is a simplified model - actual usage may vary
        
        # Input embeddings
        input_memory = batch_size * sequence_length * d_model * 4  # 4 bytes per float32
        
        # Attention matrices (Q, K, V for each layer)
        attention_memory = num_layers * batch_size * sequence_length * sequence_length * 4
        
        # Feed-forward layers (assuming 4x expansion)
        ff_memory = num_layers * batch_size * sequence_length * d_model * 4 * 4
        
        # Gradients (roughly 2x the forward pass)
        total_forward = input_memory + attention_memory + ff_memory
        total_with_gradients = total_forward * 2
        
        return total_with_gradients / (1024**3)  # Convert to GB
    
    def suggest_batch_configuration(self, target_batch_size: int, 
                                  sequence_length: int, d_model: int, 
                                  num_layers: int) -> Dict[str, Any]:
        """
        Suggest optimal batch configuration based on memory constraints.
        """
        # Get available memory
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            available_memory = total_memory * 0.8  # Leave 20% buffer
        else:
            available_memory = 8.0  # Assume 8GB for CPU
        
        # Estimate memory for target configuration
        estimated_memory = self.estimate_memory_usage(
            target_batch_size, sequence_length, d_model, num_layers
        )
        
        suggestions = {
            'target_batch_size': target_batch_size,
            'estimated_memory_gb': estimated_memory,
            'available_memory_gb': available_memory,
            'memory_fits': estimated_memory <= available_memory,
            'optimizations': []
        }
        
        if estimated_memory > available_memory:
            # Suggest optimizations
            suggestions['optimizations'].append("Enable gradient checkpointing")
            
            # Suggest smaller batch size
            max_safe_batch = int(target_batch_size * (available_memory / estimated_memory))
            suggestions['suggested_batch_size'] = max(1, max_safe_batch)
            suggestions['optimizations'].append(f"Reduce batch size to {suggestions['suggested_batch_size']}")
            
            # Suggest chunked processing
            if sequence_length > 64:
                suggestions['optimizations'].append("Use chunked attention for long sequences")
        
        return suggestions
    
    def process_variable_length_batch(
        self,
        model_forward_fn,
        inputs: list,
        max_memory_gb: float = 8.0
    ) -> list:
        """
        Process a batch with variable sequence lengths efficiently.
        """
        sorted_inputs = self._sort_inputs_by_length(inputs)
        outputs = [None] * len(inputs)
        current_batch: list = []
        current_indices: list = []

        for original_idx, input_tensor in sorted_inputs:
            current_batch.append(input_tensor)
            current_indices.append(original_idx)

            if self._should_flush_batch(current_batch, max_memory_gb):
                self._process_and_store_batch(
                    model_forward_fn,
                    current_batch,
                    current_indices,
                    outputs
                )
                current_batch, current_indices = [], []

        if current_batch:
            self._process_and_store_batch(
                model_forward_fn,
                current_batch,
                current_indices,
                outputs
            )

        return outputs

    def _sort_inputs_by_length(self, inputs: list) -> list:
        return sorted(
            enumerate(inputs),
            key=lambda item: self._sequence_length(item[1])
        )

    def _should_flush_batch(self, current_batch: list, max_memory_gb: float) -> bool:
        if not current_batch:
            return False
        if len(current_batch) >= self.max_batch_size:
            return True
        estimated_memory = self._estimate_batch_memory_gb(current_batch)
        return estimated_memory > max_memory_gb

    def _process_and_store_batch(
        self,
        model_forward_fn,
        batch: list,
        indices: list,
        outputs: list,
    ) -> None:
        if not batch:
            return
        try:
            batch_outputs = self._run_with_monitor(model_forward_fn, batch)
            for offset, output in enumerate(batch_outputs):
                outputs[indices[offset]] = output
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                self.logger.warning(
                    f"OOM with batch size {len(batch)}, falling back to per-sample processing"
                )
                self._handle_oom_batch(model_forward_fn, batch, indices, outputs)
            else:
                raise

    def _run_with_monitor(self, model_forward_fn, batch: list) -> list:
        with self.memory_monitor.monitor_peak_memory(f"batch_size_{len(batch)}"):
            return model_forward_fn(batch)

    def _handle_oom_batch(
        self,
        model_forward_fn,
        batch: list,
        indices: list,
        outputs: list,
    ) -> None:
        for offset, single_input in enumerate(batch):
            try:
                single_output = self._run_with_monitor(model_forward_fn, [single_input])
                outputs[indices[offset]] = single_output[0]
            except RuntimeError as exc:
                self.logger.error(f"Failed to process single input: {exc}")
                raise

    def _estimate_batch_memory_gb(self, batch: list) -> float:
        if not batch:
            return 0.0
        max_seq_len = max(self._sequence_length(t) for t in batch)
        max_seq_len = min(max_seq_len, self.max_sequence_length)
        return self.estimate_memory_usage(
            len(batch),
            max_seq_len,
            self.default_d_model,
            self.default_num_layers
        )

    @staticmethod
    def _sequence_length(tensor: torch.Tensor) -> int:
        if tensor.dim() > 2:
            return tensor.shape[1]
        if tensor.dim() == 2:
            return tensor.shape[0]
        return 1


def optimize_memory_usage():
    """
    Apply global memory optimizations.
    """
    # Enable memory efficient attention if available
    try:
        torch.backends.cuda.enable_flash_sdp(True)
    except AttributeError:
        pass
    
    # Set memory fraction for CUDA
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)
    
    # Enable garbage collection optimizations
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_memory_optimization_config(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate memory optimization configuration based on model parameters.
    
    Args:
        model_config: DiT model configuration
        
    Returns:
        Dictionary with memory optimization settings
    """
    d_model = model_config.get('d_model', 512)
    num_layers = model_config.get('num_layers', 12)
    max_seq_len = model_config.get('max_sequence_length', 100)
    
    # Determine if gradient checkpointing should be enabled
    # Consider both model size and sequence length
    model_params = d_model * d_model * num_layers * 4  # Rough parameter count estimate
    model_size_mb = model_params / (1024 * 1024)  # Convert to MB
    
    # Enable checkpointing for large models or long sequences
    enable_checkpointing = (
        model_size_mb > 50 or  # Models > 50MB
        (d_model >= 1024) or   # Large embedding dimension
        (num_layers >= 12) or  # Deep models
        (max_seq_len >= 200)   # Long sequences
    )
    
    # Determine chunk size for attention
    if max_seq_len > 512:
        chunk_size = 256
    elif max_seq_len > 256:
        chunk_size = 128
    else:
        chunk_size = max_seq_len
    
    return {
        'gradient_checkpointing': enable_checkpointing,
        'attention_chunk_size': chunk_size,
        'use_flash_attention': torch.cuda.is_available(),
        'max_batch_size': 16 if enable_checkpointing else 32,
        'memory_monitoring': True
    }
