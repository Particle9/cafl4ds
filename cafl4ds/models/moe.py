import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from cafl4ds.models.vit import Mlp, Block, TinyViTEncoder

class MoEMlp(nn.Module):
    """Mixture of Experts layer replacing a standard MLP."""
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int = 4,
        top_k: int = 1,
        routing_level: str = "token",
        balancing_mode: str = "batch",
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.routing_level = routing_level
        self.balancing_mode = balancing_mode
        self.dim = dim

        self.experts = nn.ModuleList([Mlp(dim, hidden_dim) for _ in range(num_experts)])
        self.router = nn.Linear(dim, num_experts)
        
        self.register_buffer("ema_expert_load", torch.ones(num_experts) / num_experts)
        self.register_buffer("expert_bias", torch.zeros(num_experts))
        
        self.aux_loss = 0.0
        self.router_probs = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        
        if self.routing_level == "image":
            router_input = x.mean(dim=1, keepdim=True)
        else:
            router_input = x
            
        router_logits = self.router(router_input)
        
        if self.balancing_mode == "bias":
            router_logits = router_logits + self.expert_bias
            
        router_probs = F.softmax(router_logits, dim=-1)
        
        if self.routing_level == "image":
            self.router_probs = router_probs.expand(b, t, self.num_experts)
        else:
            self.router_probs = router_probs
            
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.top_k > 1:
            top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)
        
        # Auxiliary loss computation
        self.aux_loss = 0.0
        if self.training:
            flat_probs = router_probs.reshape(-1, self.num_experts)
            expert_importance = flat_probs.mean(dim=0)
            
            flat_indices = top_k_indices.reshape(-1, self.top_k)
            one_hot = F.one_hot(flat_indices, num_classes=self.num_experts).float()
            expert_load = one_hot.sum(dim=1).mean(dim=0)
            
            if self.balancing_mode in ["batch", "ema"]:
                if self.balancing_mode == "ema":
                    alpha = 0.9
                    self.ema_expert_load.mul_(alpha).add_(expert_load.detach() * (1 - alpha))
                    target_load = self.ema_expert_load
                else:
                    target_load = expert_load
                
                importance_loss = self.num_experts * (expert_importance ** 2).sum()
                load_loss = self.num_experts * (target_load ** 2).sum()
                self.aux_loss = importance_loss + load_loss
                
            elif self.balancing_mode == "bias":
                alpha = 0.9
                self.ema_expert_load.mul_(alpha).add_(expert_load.detach() * (1 - alpha))
                target_load = self.ema_expert_load / (self.ema_expert_load.sum() + 1e-9)
                self.expert_bias.sub_(0.1 * (target_load - 1.0 / self.num_experts))
        
        # Dense dispatch
        expert_outputs = []
        for i in range(self.num_experts):
            expert_outputs.append(self.experts[i](x))
        expert_outputs = torch.stack(expert_outputs, dim=-1)
        
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            indices = top_k_indices[..., k]
            probs = top_k_probs[..., k]
            
            if self.routing_level == "image":
                indices = indices.expand(b, t)
                probs = probs.expand(b, t)
                
            indices_expanded = indices.unsqueeze(-1).expand(-1, -1, d)
            
            flat_outputs = expert_outputs.reshape(-1, self.num_experts)
            flat_indices = indices_expanded.reshape(-1, 1)
            selected = flat_outputs.gather(1, flat_indices).reshape(b, t, d)
            
            output += selected * probs.unsqueeze(-1)
            
        return output


class MoEBlock(Block):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        num_experts: int,
        top_k: int,
        routing_level: str,
        balancing_mode: str
    ):
        super().__init__(dim, num_heads, mlp_ratio)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = MoEMlp(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            routing_level=routing_level,
            balancing_mode=balancing_mode
        )


class MoETinyViTEncoder(TinyViTEncoder):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 8,
        in_chans: int = 3,
        embed_dim: int = 96,
        depth: int = 4,
        num_heads: int = 3,
        mlp_ratio: float = 2.0,
        num_experts: int = 4,
        top_k: int = 1,
        moe_blocks: Tuple[int, ...] = (2, 3),
        routing_level: str = "token",
        balancing_mode: str = "batch",
        cross_view_consistency: bool = False,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.moe_blocks = moe_blocks
        self.cross_view_consistency = cross_view_consistency
        
        for i in self.moe_blocks:
            self.blocks[i] = MoEBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                num_experts=num_experts,
                top_k=top_k,
                routing_level=routing_level,
                balancing_mode=balancing_mode
            )
            
    @property
    def aux_loss(self) -> torch.Tensor:
        loss = 0.0
        for i in self.moe_blocks:
            if hasattr(self.blocks[i].mlp, "aux_loss"):
                loss = loss + self.blocks[i].mlp.aux_loss
        return loss

    @torch.no_grad()
    def routing(self, imgs: torch.Tensor) -> list[torch.Tensor]:
        imgs = imgs.to(self.pos_embed.device)
        self.forward(imgs)
        
        probs = []
        for i in self.moe_blocks:
            if hasattr(self.blocks[i].mlp, "router_probs") and self.blocks[i].mlp.router_probs is not None:
                probs.append(self.blocks[i].mlp.router_probs)
        return probs
