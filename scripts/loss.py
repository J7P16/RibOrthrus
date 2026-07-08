import torch
from torch import Tensor
from jaxtyping import Bool, Float

def _safe_masked_mean(x: Float[Tensor, "*dims"], mask: Bool[Tensor, "#*dims"] | None = None) -> Float[Tensor, ""]:
    """Mean with masking that stays finite when everything is masked."""
    if mask is None:
        mask_f = torch.ones_like(x, dtype=torch.float32)
        masked = x
    else:
        mask_f = torch.broadcast_to(mask, x.shape).to(x.device, dtype=torch.float32)
        masked = x * mask_f

    masked = masked.to(torch.float32)
    denom = mask_f.sum(dtype=torch.float32).clamp(min=1.0)
    return masked.sum(dtype=torch.float32) / denom


def poisson_loss(*, y_true: Float[Tensor, "*dims"], y_pred: Float[Tensor, "*dims"], mask: Bool[Tensor, "#*dims"] | None = None) -> Float[Tensor, ""]:
    """Poisson loss implemented in torch."""
    y_true = torch.abs(y_true).to(torch.float32)
    y_pred = y_pred.to(torch.float32)

    y_pred_logits = torch.log(y_pred + 1e-7)
    min_value = y_true - y_true * torch.log(y_true + 1e-7)
    loss = (y_pred - y_true * y_pred_logits) - min_value
    return _safe_masked_mean(loss, mask)


def multinomial_loss(*, y_true: Float[Tensor, "... 1 d"], y_pred: Float[Tensor, "... 1 d"], mask: Bool[Tensor, "... 1 d"] | None = None, multinomial_resolution: int, positional_weight: float) -> dict[str, Tensor]:
    """Multinomial + Poisson loss for count predictions using torch."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape {y_true.shape} doesnot match y_pred shape {y_pred.shape}.")
    if y_pred.shape[-2] % multinomial_resolution != 0:
        raise ValueError(f"y_pred.shape[-2]={y_pred.shape[-2]} must be divisible by multinomial_resolution={multinomial_resolution}.")

    num_segments = y_pred.shape[-2] // multinomial_resolution

    y_true = torch.clamp(y_true, min=0).to(torch.float32)
    y_pred = y_pred.to(torch.float32)
    mask = mask.to(device=y_true.device)
    
    if mask.ndim == 2:
        mask = mask.unsqueeze(-1)

    y_true = y_true * mask
    y_pred = y_pred * mask

    new_shape = (
            *y_true.shape[:-2],
            num_segments,
            multinomial_resolution,
            y_true.shape[-1],
    )
    y_true = y_true.reshape(new_shape)
    y_pred = y_pred.reshape(new_shape)

    total_pred = y_pred.sum(dim=-2, keepdim=True, dtype=torch.float32)
    total_true = y_true.sum(dim=-2, keepdim=True, dtype=torch.float32)
    #mask_expanded = mask[..., None, :]
    mask_expanded = mask.reshape(
            *mask.shape[:-2],
            num_segments,
            multinomial_resolution,
            mask.shape[-1]
    ).any(dim=-2, keepdim=True)

    loss_total_count = poisson_loss(
            y_true=total_true,
            y_pred=total_pred,
            mask=mask_expanded,
    )
    loss_total_count = loss_total_count / float(multinomial_resolution)

    prob_predictions = y_pred / (total_pred + 1e-7)
    loss_positional = -y_true * torch.log(prob_predictions + 1e-7)
    loss_positional = _safe_masked_mean(loss_positional, mask=mask_expanded)

    return {
        "loss": loss_total_count + positional_weight * loss_positional, 
        "loss_total": loss_total_count,
        "loss_positional": loss_positional,
        "max_sum_preds": torch.max(total_pred),
        "max_sum_targets": torch.max(total_true),
        "max_preds": torch.max(y_pred),
        "max_targets": torch.max(y_true).to(torch.float32)
    }

#test = multinomial_loss(
#        y_true=torch.poisson(torch.ones(32, 128, 1) * 2.0), 
#        y_pred=torch.rand(32, 128, 1) * 3.0, 
#        mask=torch.ones(32, 128, 1, dtype=torch.bool),
#        multinomial_resolution=4,
#        positional_weight=0.281
#)
#print(test)
