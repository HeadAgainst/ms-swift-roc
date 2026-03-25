# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.nn.functional as F

from .base import BaseLoss


class CustomCrossEntropyLoss(BaseLoss):

    def __call__(self, outputs, labels, *, num_items_in_batch=None, loss_scale=None, **kwargs):
        from swift.trainers import per_token_loss_func
        token_loss = per_token_loss_func(outputs, labels)
        if num_items_in_batch is None:
            num_items_in_batch = (labels[:, 1:] != -100).sum()
        return token_loss.sum() / num_items_in_batch


class RocScoreL1Loss(BaseLoss):

    def __call__(self, outputs, labels, *, num_items_in_batch=None, gt_score=None, **kwargs):
        from swift.trainers import per_token_loss_func

        mode = 'train' if self.trainer.model.training else 'eval'
        token_loss = per_token_loss_func(outputs, labels)
        if num_items_in_batch is None:
            num_items_in_batch = (labels[:, 1:] != -100).sum()
        lm_ce_loss = token_loss.sum() / num_items_in_batch
        self.trainer.custom_metrics[mode]['lm_ce_loss'].update(lm_ce_loss.detach())
        self.trainer.forced_log_scalars[mode]['lm_ce_loss'] = float(lm_ce_loss.detach().float().item())
        if gt_score is None:
            return lm_ce_loss

        tokenizer = self.trainer.template.tokenizer
        score_token = self.args.roc_score_token
        score_token_id = tokenizer.convert_tokens_to_ids(score_token)
        bucket_token_ids = tokenizer.convert_tokens_to_ids(
            [self.args.roc_bucket_token_template.format(i) for i in range(self.args.roc_num_tokens)])

        if score_token_id == tokenizer.unk_token_id and score_token != tokenizer.unk_token:
            raise ValueError(f'ROC score token `{score_token}` is not found in tokenizer.')
        if any(token_id == tokenizer.unk_token_id for token_id in bucket_token_ids if tokenizer.unk_token_id is not None):
            raise ValueError('Some ROC bucket tokens are not found in tokenizer.')

        score_positions = (labels == score_token_id).nonzero(as_tuple=False)
        if score_positions.numel() == 0:
            return lm_ce_loss

        valid_mask = score_positions[:, 1] > 0
        score_positions = score_positions[valid_mask]
        if score_positions.numel() == 0:
            return lm_ce_loss

        batch_indices = score_positions[:, 0]
        logit_indices = score_positions[:, 1] - 1
        score_logits = outputs.logits[batch_indices, logit_indices][:, bucket_token_ids].float()
        score_probs = torch.softmax(score_logits, dim=-1)
        score_weights = torch.linspace(
            self.args.roc_min_score, self.args.roc_max_score, steps=self.args.roc_num_tokens, device=score_probs.device)
        pred_score = (score_probs * score_weights).sum(dim=-1)

        if not isinstance(gt_score, torch.Tensor):
            gt_score = torch.tensor(gt_score, device=pred_score.device, dtype=pred_score.dtype)
        gt_score = gt_score.to(device=pred_score.device, dtype=pred_score.dtype).view(-1)
        gt_score = gt_score[batch_indices]

        score_span = self.args.roc_max_score - self.args.roc_min_score
        if score_span <= 0:
            raise ValueError('`roc_max_score` must be greater than `roc_min_score`.')
        target_bucket = torch.round(
            (gt_score - self.args.roc_min_score) / score_span * (self.args.roc_num_tokens - 1)).long()
        target_bucket = target_bucket.clamp_(0, self.args.roc_num_tokens - 1)
        bucket_ce_loss = F.cross_entropy(score_logits, target_bucket)

        l1_loss = F.smooth_l1_loss(pred_score, gt_score)
        self.trainer.custom_metrics[mode]['bucket_ce_loss'].update(bucket_ce_loss.detach())
        self.trainer.custom_metrics[mode]['pred_score_l1'].update(l1_loss.detach())
        self.trainer.custom_metrics[mode]['pred_score_mean'].update(pred_score.detach())
        self.trainer.custom_metrics[mode]['target_bucket_mean'].update(target_bucket.detach().float())
        self.trainer.forced_log_scalars[mode]['bucket_ce_loss'] = float(bucket_ce_loss.detach().float().item())
        self.trainer.forced_log_scalars[mode]['pred_score_l1'] = float(l1_loss.detach().float().item())
        self.trainer.forced_log_scalars[mode]['pred_score_mean'] = float(pred_score.detach().float().mean().item())
        self.trainer.forced_log_scalars[mode]['target_bucket_mean'] = float(
            target_bucket.detach().float().mean().item())
        return lm_ce_loss + self.args.roc_bucket_ce_weight * bucket_ce_loss + self.args.roc_l1_weight * l1_loss
