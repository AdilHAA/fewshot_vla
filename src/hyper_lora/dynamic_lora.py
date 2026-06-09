import torch
import torch.nn as nn


class DynamicLoRALinear(nn.Module):
    """
    Wrapper around a frozen `nn.Linear` whose LoRA matrices are produced by a
    hypernetwork on every forward pass instead of being learned as own
    parameters.

    Forward:
        y = x W^T + scaling * (x W_down^T) W_up^T + bias
    where (W_down, W_up) are set externally via `set_lora_weights` before the
    parent module's `forward` is invoked.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        lora_rank: int = 4,
        lora_alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.lora_rank

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        self.current_w_down: torch.Tensor | None = None
        self.current_w_up: torch.Tensor | None = None

    def set_lora_weights(self, w_down: torch.Tensor, w_up: torch.Tensor) -> None:
        self.current_w_down = w_down
        self.current_w_up = w_up

    def clear_lora_weights(self) -> None:
        self.current_w_down = None
        self.current_w_up = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        if self.current_w_down is None or self.current_w_up is None:
            return base_out

        x_drop = self.dropout(x)
        w_d = self.current_w_down.to(x.dtype)
        w_u = self.current_w_up.to(x.dtype)

        # x: (..., in)  -> reshape leading dims to batch B
        leading = x_drop.shape[:-1]
        B = w_d.shape[0]
        x_flat = x_drop.reshape(B, -1, x_drop.shape[-1])  # (B, T, in)

        x_down = torch.bmm(x_flat, w_d.transpose(1, 2))   # (B, T, r)
        lora_out = torch.bmm(x_down, w_u.transpose(1, 2)) # (B, T, out)
        lora_out = lora_out.reshape(*leading, self.out_features)
        return base_out + lora_out * self.scaling
