import torch
from torch import nn

from hurricore.models.swi_glu import SwiGLU


class Expert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.swish_glu = SwiGLU(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.swish_glu(x)
        output = self.output_layer(hidden)
        return output


class Gate(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        num_experts: int = 64,
        top_k: int = 6,
        bias_step_size: float = 0.001,
    ):
        super().__init__()
        assert top_k <= num_experts, f"top_k ({top_k}) must be less than or equal to num_experts ({num_experts})"
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.w = nn.Linear(input_dim, num_experts)
        self.routing_bias = nn.Parameter(torch.zeros(1, 1, num_experts))
        self.bias_step_size = bias_step_size
    

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits: torch.Tensor = self.w(x)  # (batch_size, num_tokens, num_experts)
        weights = logits.sigmoid() + self.routing_bias
        top_k_indices = torch.topk(weights, k=self.top_k, dim=-1)[1]  # (batch_size, num_tokens, top_k)
        top_k_weights = torch.gather(weights, dim=-1, index=top_k_indices)  # (batch_size, num_tokens, top_k)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        # Update routing bias when training
        if self.training:
            self._update_routing_bias(x, top_k_indices)
        return top_k_indices, top_k_weights
    

    @torch.no_grad
    @torch.jit.ignore
    def _update_routing_bias(self, x, top_k_indices):
        # Count the number of assigned tokens c_i to each expert, and the average number c_mean
        batch_size, num_tokens = x.shape[0], x.shape[1]
        mask = torch.zeros(batch_size, num_tokens, self.num_experts, device=x.device)
        mask.scatter_(-1, top_k_indices, 1)  # Set 1s at positions of selected experts
        # Sum across batch and tokens to get count per expert
        token_cnt_per_expert = mask.sum(dim=[0, 1])  # Shape: (num_experts,)
        # Calculate mean usage across experts
        mean_token_cnt = token_cnt_per_expert.mean()
        load_violation_error = (token_cnt_per_expert - mean_token_cnt).sign().expand_as(self.routing_bias.data)
        self.routing_bias.data -= load_violation_error * self.bias_step_size



class MoE(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        expert_dim: int = 128,
        num_shared_experts: int = 2,
        num_routed_experts: int = 64,
        top_k: int = 6,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.expert_dim = expert_dim
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.gate = Gate(input_dim, num_routed_experts, top_k)
        self.shared_experts = Expert(input_dim, num_shared_experts * expert_dim)
        self.routed_experts = nn.ModuleList(
            [
                Expert(input_dim, expert_dim) 
                for _ in range(num_routed_experts)
            ]
        )
    
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Highly optimized forward pass using torch.where for efficient grouping.
        """
        batch_size, num_tokens, _ = x.shape
        
        # Get shared expert output
        shared_output = self.shared_experts(x)
        
        # Get routing information
        top_k_indices, top_k_weights = self.gate(x)  
        
        # Reshape input for processing
        x_flat = x.view(-1, self.input_dim)       
        output_flat = torch.zeros_like(x_flat)    
        
        # Flatten expert indices
        indices = top_k_indices.view(-1)              
        
        # Create source indices to track which token each expert index belongs to
        source_indices = torch.arange(batch_size * num_tokens, device=x.device).repeat_interleave(self.top_k)
        
        # Flatten weights
        weights = top_k_weights.view(-1)              
        
        # Count number of tokens per expert for efficient processing
        counts = torch.zeros(self.num_routed_experts, dtype=torch.long, device=x.device)
        counts.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.long))
        # Process each expert
        for expert_idx, expert in enumerate(self.routed_experts):
            if counts[expert_idx] > 0:
                # Find positions where this expert is selected
                token_positions = torch.where(indices == expert_idx)[0]  # Get the first (and only) tensor of indices
                
                # Get the corresponding token indices and weights
                token_indices = source_indices[token_positions]
                token_weights = weights[token_positions]
                
                # Process tokens with this expert
                expert_outputs = expert(x_flat[token_indices])
                
                # Apply weights
                weighted_outputs = expert_outputs * token_weights.unsqueeze(1)
                
                # Add to the output tensor
                for i, idx in enumerate(token_indices):
                    output_flat[idx] += weighted_outputs[i]
        
        # Reshape back to original dimensions
        output = output_flat.view(batch_size, num_tokens, -1)
        
        return output + shared_output
