"""
nanoMAMBA - Mamba2 model wrapper for nanochat
==============================================

This module wraps the official mamba-ssm implementation to match nanochat's API.
We use the optimized CUDA kernels from the official repo for maximum training speed!

Installation:
    pip install mamba-ssm --break-system-packages
    
Reference:
    - Official Mamba: https://github.com/state-spaces/mamba
    - Mamba-2 paper: https://arxiv.org/abs/2405.21060
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

# Import from official mamba-ssm package (will be installed)
try:
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel as OfficialMambaLM
    from mamba_ssm import Mamba2
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("WARNING: mamba-ssm not installed. Install with: pip install mamba-ssm")
    

@dataclass
class MambaConfig:
    """Configuration for nanoMAMBA model.
    
    Maps nanochat's GPT config parameters to Mamba equivalents.
    """
    # Core model dimensions
    vocab_size: int = 65536  # nanochat's BPE tokenizer vocab size
    d_model: int = 1280      # hidden dimension (was n_embd in GPT)
    n_layer: int = 20        # number of Mamba blocks (was n_layer in GPT) 
    
    # Mamba-specific parameters
    d_state: int = 128       # SSM state dimension (N in paper)
    d_conv: int = 4          # convolution kernel size
    expand: int = 2          # expansion factor (E in paper)
    headdim: int = 64        # head dimension for Mamba2
    chunk_size: int = 256    # chunk size for SSD algorithm
    
    # Training parameters
    bias: bool = False       # whether to use bias in linear layers
    dropout: float = 0.0     # dropout (Mamba typically doesn't use dropout)
    
    # Computed properties
    def __post_init__(self):
        self.d_inner = self.expand * self.d_model
        

class Mamba(nn.Module):
    """
    Mamba Language Model - Drop-in replacement for nanochat's GPT.
    
    This wraps the official mamba-ssm implementation with nanochat's API.
    Uses optimized CUDA kernels for fast training!
    """
    
    def __init__(self, config: MambaConfig):
        super().__init__()
        assert MAMBA_AVAILABLE, "mamba-ssm package required! Install: pip install mamba-ssm"
        self.config = config
        
        # Build the Mamba model using official implementation
        # This gets us all the optimized CUDA kernels!
        self.model = OfficialMambaLM(
            d_model=config.d_model,
            n_layer=config.n_layer,
            d_state=config.d_state,
            d_conv=config.d_conv, 
            expand=config.expand,
            vocab_size=config.vocab_size,
            # Mamba2-specific params
            ssm_cfg=dict(
                headdim=config.headdim,
                chunk_size=config.chunk_size,
                layer='Mamba2',  # Use Mamba2 architecture
            ),
            rms_norm=True,
            fused_add_norm=True,  # Use fused operations for speed
            residual_in_fp32=True,
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report parameter count
        n_params = sum(p.numel() for p in self.parameters())
        print(f"nanoMAMBA model initialized: {n_params:,} parameters")
        
    def _init_weights(self, module):
        """Initialize weights similar to nanochat's GPT."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None, return_logits=True):
        """
        Forward pass matching nanochat's GPT API.
        
        Args:
            idx: (batch, seqlen) input token indices
            targets: (batch, seqlen) target token indices (for training)
            return_logits: whether to return logits (compatibility with nanochat)
            
        Returns:
            if targets is None:
                logits: (batch, seqlen, vocab_size)
            else:
                (loss, logits) or just loss depending on return_logits
        """
        # Forward through Mamba model
        logits = self.model(idx)
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                ignore_index=-1
            )
        
        # Return based on what's requested (match nanochat API)
        if targets is None:
            return logits
        elif return_logits:
            return loss, logits
        else:
            return loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate tokens autoregressively.
        
        Args:
            idx: (batch, seqlen) input context
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling (if specified)
            
        Returns:
            idx: (batch, seqlen + max_new_tokens) generated sequence
        """
        for _ in range(max_new_tokens):
            # Forward pass
            logits = self(idx)
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
    
    @torch.no_grad()
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        Estimate model flops utilization (MFU).
        
        This is a rough estimate adapted from nanochat's GPT implementation.
        Mamba has different FLOPs profile than Transformers, but this gives
        a ballpark for hardware utilization.
        """
        # Rough estimate: Mamba has ~6*N*D FLOPs per token 
        # (much less than Transformer's ~12*N*D due to no attention)
        N = self.config.d_model
        L = self.config.n_layer
        flops_per_token = 6 * N * L
        flops_per_fwdbwd = flops_per_token * fwdbwd_per_iter
        flops_per_iter = flops_per_fwdbwd
        
        # Express in terms of A100 bfloat16 peak FLOPs
        flops_achieved = flops_per_iter / dt
        flops_promised = 312e12  # A100 bfloat16 peak
        mfu = flops_achieved / flops_promised
        return mfu
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure optimizer groups - matches nanochat API.
        
        For Mamba, we follow similar logic: weight decay on matmuls, no decay on norms/biases.
        """
        # Separate parameters that should and shouldn't have weight decay
        decay = set()
        no_decay = set()
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn
                
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, (nn.Linear,)):
                    decay.add(fpn)
                elif 'norm' in mn.lower():
                    no_decay.add(fpn)
                else:
                    # SSM parameters - typically don't decay
                    no_decay.add(fpn)
        
        # Create optimizer groups
        param_dict = {pn: p for pn, p in self.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        
        # Use AdamW
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer


def create_model(config):
    """Factory function to create Mamba model - matches nanochat interface."""
    return Mamba(config)


if __name__ == "__main__":
    # Quick test
    print("Testing nanoMAMBA model...")
    
    config = MambaConfig(
        vocab_size=65536,
        d_model=256,  # small for testing
        n_layer=4,
        d_state=64,
    )
    
    model = Mamba(config)
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits = model(idx)
    print(f"Input shape: {idx.shape}")
    print(f"Output shape: {logits.shape}")
    print("✅ nanoMAMBA model working!")
