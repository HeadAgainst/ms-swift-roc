# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.nn.functional as F
from accelerate.utils import extract_model_from_parallel
from peft import PeftModel

from .base import BaseLoss


class CustomCrossEntropyLoss(BaseLoss):

    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs):
        from swift.trainers import per_token_loss_func
        token_loss = per_token_loss_func(outputs, labels)
        if num_items_in_batch is None:
            num_items_in_batch = (labels[:, 1:] != -100).sum()
        return token_loss.sum() / num_items_in_batch


class RocScoreL1Loss(BaseLoss):

    def _get_roc_attr(self, name: str):
        if hasattr(self.args, name):
            return getattr(self.args, name)
        model_config = getattr(self.trainer.model, 'config', None)
        if model_config is not None and hasattr(model_config, name):
            return getattr(model_config, name)
        raise AttributeError(f'ROC attribute `{name}` is not found in training args or model.config.')

    def __call__(self, outputs, labels, *, num_items_in_batch=None, gt_score=None, lengths=None, **kwargs):
        from swift.trainers import per_token_loss_func
        from swift.model.utils import get_roc_score_head

        mode = 'train' if self.trainer.model.training else 'eval'
        token_loss = per_token_loss_func(outputs, labels)
        if num_items_in_batch is None:
            num_items_in_batch = (labels[:, 1:] != -100).sum()
        lm_ce_loss = token_loss.sum() / num_items_in_batch
        self.trainer.custom_metrics[mode]['lm_ce_loss'].update(lm_ce_loss.detach())
        self.trainer.forced_log_scalars[mode]['lm_ce_loss'] = float(lm_ce_loss.detach().float().item())
        if gt_score is None:
            return lm_ce_loss

        score_token = self._get_roc_attr('roc_score_token')
        roc_num_tokens = self._get_roc_attr('roc_num_tokens')
        roc_min_score = self._get_roc_attr('roc_min_score')
        roc_max_score = self._get_roc_attr('roc_max_score')
        roc_l1_weight = self._get_roc_attr('roc_l1_weight')
        tokenizer = self.trainer.template.tokenizer
        score_token_id = tokenizer.convert_tokens_to_ids(score_token)
        if score_token_id is None:
            raise ValueError(f'ROC score token `{score_token}` is not found in tokenizer.')
        if tokenizer.unk_token_id is not None and score_token_id == tokenizer.unk_token_id and score_token != tokenizer.unk_token:
            raise ValueError(f'ROC score token `{score_token}` is not found in tokenizer.')

        score_positions = (labels == score_token_id).nonzero(as_tuple=False)
        if score_positions.numel() == 0:
            return lm_ce_loss

        valid_mask = score_positions[:, 1] > 0
        score_positions = score_positions[valid_mask]
        if score_positions.numel() == 0:
            return lm_ce_loss

        batch_indices = score_positions[:, 0]
        hidden_indices = score_positions[:, 1] - 1
        hidden_states = getattr(outputs, 'hidden_states', None)
        if hidden_states is None and isinstance(outputs, dict):
            hidden_states = outputs.get('hidden_states')
        if hidden_states is None:
            raise ValueError('ROC score head requires `output_hidden_states=True`, but hidden states are missing.')
        last_hidden_state = hidden_states[-1]

        model = extract_model_from_parallel(self.trainer.model)
        if isinstance(model, PeftModel):
            model = model.model
        score_head = get_roc_score_head(model)
        if score_head is None:
            raise ValueError('ROC score head is missing on the model.')
        score_inputs = last_hidden_state[batch_indices, hidden_indices].to(
            device=score_head.weight.device, dtype=score_head.weight.dtype)
        score_logits = score_head(score_inputs).float()
        score_probs = torch.softmax(score_logits, dim=-1)
        score_weights = torch.linspace(
            roc_min_score, roc_max_score, steps=roc_num_tokens, device=score_probs.device, dtype=score_probs.dtype)
        pred_score = (score_probs * score_weights).sum(dim=-1)

        if not isinstance(gt_score, torch.Tensor):
            gt_score = torch.tensor(gt_score, device=pred_score.device, dtype=pred_score.dtype)
        gt_score = gt_score.to(device=pred_score.device, dtype=pred_score.dtype).view(-1)
        if lengths is not None:
            if isinstance(lengths, torch.Tensor):
                lengths = lengths.detach().cpu().tolist()
            flat_lengths = []
            for value in lengths:
                if isinstance(value, (list, tuple)):
                    if len(value) != 1:
                        raise ValueError(f'Unexpected packed lengths item: {value}')
                    value = value[0]
                flat_lengths.append(int(value))
            if len(flat_lengths) == gt_score.shape[0] and gt_score.shape[0] > 1:
                cum_lengths = torch.tensor(flat_lengths, device=labels.device).cumsum(dim=0)
                sample_indices = torch.searchsorted(cum_lengths, score_positions[:, 1], right=True)
                gt_score = gt_score[sample_indices]
            else:
                gt_score = gt_score[batch_indices]
        else:
            gt_score = gt_score[batch_indices]

        l1_loss = F.smooth_l1_loss(pred_score, gt_score)
        self.trainer.custom_metrics[mode]['pred_score_l1'].update(l1_loss.detach())
        self.trainer.custom_metrics[mode]['pred_score_mean'].update(pred_score.detach())
        self.trainer.forced_log_scalars[mode]['pred_score_l1'] = float(l1_loss.detach().float().item())
        self.trainer.forced_log_scalars[mode]['pred_score_mean'] = float(pred_score.detach().float().mean().item())
        return lm_ce_loss + roc_l1_weight * l1_loss
