"""AdaRCG model implementation used by the four public dataset pipelines."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class AdaRCG(nn.Module):
    def __init__(
        self,
        membership: torch.Tensor,
        region_adjacency: torch.Tensor,
        input_dim: int,
        classes: int,
        hidden_dim: int = 48,
        rank: int = 4,
        top_regions: int = 2,
        dropout: float = 0.15,
        auxiliary_weight: float = 0.0,
        fusion_mode: str = "reliability",
        structured_evidence: bool = False,
    ) -> None:
        super().__init__()
        if membership.ndim != 2:
            raise ValueError("membership must be region x channel")
        regions, channels = membership.shape
        if region_adjacency.shape != (regions, regions):
            raise ValueError("region adjacency shape mismatch")
        self.channels = channels
        self.regions = regions
        self.classes = classes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.top_regions = min(top_regions, regions)
        self.auxiliary_weight = auxiliary_weight
        self.structured_evidence = bool(structured_evidence)
        if fusion_mode not in {"reliability", "uniform"}:
            raise ValueError(f"unknown fusion mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.register_buffer("membership", membership.float())
        self.register_buffer("region_prior", region_adjacency.float())

        self.temporal_mode = input_dim > 15 and input_dim % 5 == 0
        if self.temporal_mode:
            self.band_encoder = nn.Sequential(
                nn.LayerNorm(5),
                nn.Linear(5, hidden_dim),
                nn.GELU(),
            )
            self.temporal_depthwise = nn.Conv1d(
                hidden_dim, hidden_dim, kernel_size=3, padding=1,
                groups=hidden_dim,
            )
            self.temporal_mix = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
            self.temporal_score = nn.Linear(hidden_dim, 1)
            self.temporal_output_depthwise = nn.Conv1d(
                hidden_dim, hidden_dim, kernel_size=3, padding=1,
                groups=hidden_dim,
            )
            self.temporal_output_mix = nn.Conv1d(
                hidden_dim, hidden_dim, kernel_size=1
            )
            self.temporal_output_score = nn.Linear(hidden_dim, 1)
            # Low-variance spectral evidence anchors the adaptive graph branch
            # when subject-specific training data are limited.
            self.stable_head = nn.Sequential(
                nn.LayerNorm(channels * 10),
                nn.Linear(channels * 10, classes),
            )
            self.stable_fusion_logit = nn.Parameter(torch.tensor(1.1))
            if self.structured_evidence:
                pairs = regions * (regions - 1) // 2
                connection_features = 3 * pairs * 5
                channel_features = channels * 10
                # These two linear maps are initialised jointly from the
                # training partition. Splitting the single discriminative map
                # preserves explicit channel/connection evidence while their
                # summed logits exactly reproduce the fitted anchor.
                self.stable_head = nn.Linear(channel_features, classes)
                self.connection_evidence_head = nn.Linear(
                    connection_features, classes
                )
                total_features = channel_features + connection_features
                self.register_buffer(
                    "evidence_feature_mean", torch.zeros(total_features)
                )
                self.register_buffer(
                    "evidence_feature_scale", torch.ones(total_features)
                )
                self.register_buffer(
                    "evidence_input_mean", torch.zeros(channels, 5)
                )
                self.register_buffer(
                    "evidence_input_scale", torch.ones(channels, 5)
                )
                # The adaptive graph is a bounded residual on a calibrated
                # anchor, preventing an unstable deep path from erasing the
                # cross-session evidence. Reliability modulates it per sample.
                self.decision_fusion_logits = nn.Parameter(torch.tensor(-2.0))
            else:
                self.connection_evidence_head = None
                self.decision_fusion_logits = None
            self.channel_encoder = None
        else:
            if self.structured_evidence:
                raise ValueError(
                    "structured evidence requires a temporal five-band input"
                )
            self.channel_encoder = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.stable_head = None
            self.stable_fusion_logit = None
            self.connection_evidence_head = None
            self.decision_fusion_logits = None
        # Tiny target-specific adapters. They are identity-initialized and only
        # these vectors plus the classifier are updated during calibration.
        self.personal_band_scale = nn.Parameter(torch.ones(input_dim))
        self.personal_channel_scale = nn.Parameter(torch.ones(channels))
        self.personal_region_scale = nn.Parameter(torch.ones(regions))
        self.quality_mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, max(hidden_dim // 2, 8)),
            nn.GELU(),
            nn.Linear(max(hidden_dim // 2, 8), 1),
        )
        self.signal_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.artifact_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.signal_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.artifact_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.signal_query = nn.Parameter(torch.randn(regions, hidden_dim) * 0.02)
        self.artifact_query = nn.Parameter(torch.randn(regions, hidden_dim) * 0.02)
        self.artifact_lambda = nn.Parameter(torch.full((regions,), -2.0))
        self.region_norm = nn.LayerNorm(hidden_dim)

        self.graph_left = nn.Parameter(torch.randn(regions, rank) * 0.02)
        self.graph_right = nn.Parameter(torch.randn(regions, rank) * 0.02)
        self.graph_scale = nn.Parameter(torch.tensor(-2.0))
        self.order_projections = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(3)
        )
        self.order_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 16),
            nn.GELU(),
            nn.Linear(16, 3),
        )
        self.focus_score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.glance_projection = nn.Linear(hidden_dim, hidden_dim)
        self.focus_projection = nn.Linear(hidden_dim, hidden_dim)
        self.graph_projection = nn.Linear(hidden_dim, hidden_dim)
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(3 * hidden_dim + 4),
            nn.Linear(3 * hidden_dim + 4, max(hidden_dim // 4, 16)),
            nn.GELU(),
            nn.Linear(max(hidden_dim // 4, 16), 3),
        )
        self.embedding_norm = nn.LayerNorm(hidden_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.personal_scale = nn.Parameter(torch.ones(hidden_dim))
        self.personal_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.coarse_head = nn.Linear(hidden_dim, classes)
        self.head = nn.Linear(hidden_dim, classes)
        if classes == 4:
            self.valence_head = nn.Linear(hidden_dim, 2)
            self.arousal_head = nn.Linear(hidden_dim, 2)
        else:
            self.valence_head = None
            self.arousal_head = None

    def _reliability(self, inputs: torch.Tensor) -> torch.Tensor:
        learned = torch.sigmoid(self.quality_mlp(inputs).squeeze(-1))
        # The middle third contains temporal standard deviations. Robustly
        # penalise channels whose variability departs from the sample median.
        if inputs.shape[-1] == 15:
            variability = inputs[..., 5:10].abs().mean(dim=-1)
        elif inputs.shape[-1] > 15 and inputs.shape[-1] % 5 == 0:
            sequence = inputs.reshape(*inputs.shape[:-1], -1, 5)
            variability = sequence.std(dim=-2).abs().mean(dim=-1)
        else:
            variability = inputs.abs().mean(dim=-1)
        median = variability.median(dim=1, keepdim=True).values
        deviation = (variability - median).abs()
        scale = deviation.median(dim=1, keepdim=True).values.clamp_min(0.1)
        rule = torch.exp(-deviation / (2.5 * scale)).clamp(0.05, 1.0)
        return 0.5 * learned + 0.5 * rule

    def _encode_channels(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.temporal_mode:
            return self.channel_encoder(inputs)
        batch, channels, _ = inputs.shape
        sequence = inputs.reshape(batch, channels, -1, 5)
        features = self.band_encoder(sequence)
        time = features.shape[2]
        features = features.reshape(batch * channels, time, self.hidden_dim)
        convolution_input = features.transpose(1, 2)
        residual = self.temporal_mix(
            F.gelu(self.temporal_depthwise(convolution_input))
        ).transpose(1, 2)
        features = features + residual
        weights = torch.softmax(self.temporal_score(features).squeeze(-1), dim=-1)
        pooled = torch.einsum("bt,btd->bd", weights, features)
        return pooled.reshape(batch, channels, self.hidden_dim)

    @staticmethod
    def _correlation(
        left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        """Per-band correlation across the causal within-window time axis."""

        left = left - left.mean(dim=1, keepdim=True)
        right = right - right.mean(dim=1, keepdim=True)
        numerator = (left * right).mean(dim=1)
        denominator = torch.sqrt(
            ((left.square()).mean(dim=1) + 1e-4)
            * ((right.square()).mean(dim=1) + 1e-4)
        )
        return (numerator / denominator).clamp(-1.0, 1.0)

    def _structured_connection_features(
        self, sequence: torch.Tensor
    ) -> torch.Tensor:
        """Zero-lag and directed one-step region evidence for the connector."""

        membership = self.membership / self.membership.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0)
        # sequence: batch x channel x time x band
        region = torch.einsum("rc,bctf->btrf", membership, sequence)
        pair = torch.triu_indices(
            self.regions, self.regions, offset=1, device=sequence.device
        )
        left, right = pair[0], pair[1]
        zero = self._correlation(region[:, :, left], region[:, :, right])
        forward = self._correlation(
            region[:, :-1, left], region[:, 1:, right]
        )
        backward = self._correlation(
            region[:, :-1, right], region[:, 1:, left]
        )
        return torch.cat(
            (zero.flatten(1), forward.flatten(1), backward.flatten(1)), dim=1
        )

    def structured_evidence_features(
        self, inputs: torch.Tensor
    ) -> torch.Tensor:
        """Return raw channel/region evidence in the canonical feature order."""

        if not self.structured_evidence:
            raise RuntimeError("structured evidence is disabled")
        batch, channels, _ = inputs.shape
        sequence = inputs.reshape(batch, channels, -1, 5)
        # The general neural path consumes train-normalised inputs. Reversing
        # that transform here keeps anatomical regional averages in the raw
        # DE scale used by the evidence anchor. Both buffers are fitted only
        # on the current training partition.
        sequence = (
            sequence * self.evidence_input_scale[None, :, None, :]
            + self.evidence_input_mean[None, :, None, :]
        )
        channel = torch.cat(
            (sequence.mean(dim=2), sequence.std(dim=2, unbiased=False)), dim=-1
        ).reshape(batch, -1)
        connection = self._structured_connection_features(sequence)
        return torch.cat((channel, connection), dim=1)

    @torch.no_grad()
    def set_structured_evidence_state(
        self,
        *,
        input_mean: torch.Tensor,
        input_scale: torch.Tensor,
        feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
        coefficient: torch.Tensor,
        intercept: torch.Tensor,
    ) -> None:
        """Install a train-fitted linear anchor without exposing test data."""

        if not self.structured_evidence:
            raise RuntimeError("structured evidence is disabled")
        expected = self.evidence_feature_mean.numel()
        if feature_mean.numel() != expected or feature_scale.numel() != expected:
            raise ValueError("structured feature normalisation shape mismatch")
        if coefficient.shape != (self.classes, expected):
            raise ValueError("structured coefficient shape mismatch")
        if intercept.shape != (self.classes,):
            raise ValueError("structured intercept shape mismatch")
        self.evidence_input_mean.copy_(input_mean)
        self.evidence_input_scale.copy_(input_scale)
        self.evidence_feature_mean.copy_(feature_mean)
        self.evidence_feature_scale.copy_(feature_scale.clamp_min(1e-6))
        split = self.channels * 10
        self.stable_head.weight.copy_(coefficient[:, :split])
        self.stable_head.bias.copy_(intercept)
        self.connection_evidence_head.weight.copy_(coefficient[:, split:])
        self.connection_evidence_head.bias.zero_()

    def _masked_attention(self, logits: torch.Tensor) -> torch.Tensor:
        mask = self.membership.unsqueeze(0) > 0
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        return torch.softmax(logits, dim=-1)

    def _adjacency(
        self, edge_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        residual = self.graph_left @ self.graph_right.T
        residual = 0.5 * (residual + residual.T)
        residual = torch.sigmoid(residual)
        adjacency = self.region_prior + torch.sigmoid(self.graph_scale) * residual
        adjacency = 0.5 * (adjacency + adjacency.T)
        if edge_mask is not None:
            if edge_mask.shape != adjacency.shape:
                raise ValueError(
                    f"edge mask must have shape {tuple(adjacency.shape)}, "
                    f"got {tuple(edge_mask.shape)}"
                )
            mask = edge_mask.to(
                device=adjacency.device, dtype=adjacency.dtype
            )
            if not torch.allclose(mask, mask.T):
                raise ValueError("edge mask must be symmetric")
            # Self evidence is never removed by an inter-region ablation.
            mask = mask.clone()
            mask.fill_diagonal_(1.0)
            adjacency = adjacency * mask
        degree = adjacency.sum(dim=-1).clamp_min(1e-6)
        inverse = degree.rsqrt()
        return inverse[:, None] * adjacency * inverse[None, :]

    def _focus_gate(self, graph_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.focus_score(graph_features).squeeze(-1)
        soft = torch.softmax(scores, dim=-1)
        indices = scores.topk(self.top_regions, dim=-1).indices
        hard = torch.zeros_like(soft).scatter_(-1, indices, 1.0 / self.top_regions)
        gate = hard + soft - soft.detach() if self.training else hard
        return gate, scores

    def _forward_temporal(
        self,
        inputs: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
        evidence_inputs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Fuse channels and regions at every second before temporal pooling.

        Brain connectivity may change within a ten-second window. Pooling every
        channel over time before constructing regions erases that signal, so
        the temporal path performs the lightweight regional graph block at
        each step and only then aggregates the fused evidence.
        """

        batch, channels, _ = inputs.shape
        sequence = inputs.reshape(batch, channels, -1, 5)
        channel_features = self.band_encoder(sequence)
        time = channel_features.shape[2]
        convolution_input = channel_features.reshape(
            batch * channels, time, self.hidden_dim
        ).transpose(1, 2)
        residual = self.temporal_mix(
            F.gelu(self.temporal_depthwise(convolution_input))
        ).transpose(1, 2)
        channel_features = (
            channel_features.reshape(batch * channels, time, self.hidden_dim)
            + residual
        ).reshape(batch, channels, time, self.hidden_dim)
        channel_features = channel_features.permute(0, 2, 1, 3)
        channel_features = (
            channel_features
            * self.personal_channel_scale[None, None, :, None]
        )
        reliability = self._reliability(inputs)
        scale = math.sqrt(self.hidden_dim)

        signal_logits = torch.einsum(
            "btcd,rd->btrc", self.signal_key(channel_features), self.signal_query
        ) / scale
        mask = self.membership[None, None] > 0
        signal_logits = signal_logits.masked_fill(
            ~mask, torch.finfo(signal_logits.dtype).min
        )
        signal_weights = torch.softmax(
            signal_logits + reliability[:, None, None, :].log(), dim=-1
        )
        artifact_logits = torch.einsum(
            "btcd,rd->btrc",
            self.artifact_key(channel_features), self.artifact_query,
        ) / scale
        artifact_logits = artifact_logits + 2.0 * (
            1.0 - reliability[:, None, None, :]
        )
        artifact_logits = artifact_logits.masked_fill(
            ~mask, torch.finfo(artifact_logits.dtype).min
        )
        artifact_weights = torch.softmax(artifact_logits, dim=-1)
        signal_region = torch.einsum(
            "btrc,btcd->btrd", signal_weights,
            self.signal_value(channel_features),
        )
        artifact_region = torch.einsum(
            "btrc,btcd->btrd", artifact_weights,
            self.artifact_value(channel_features),
        )
        membership = self.membership / self.membership.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        region_reliability = torch.einsum("rc,bc->br", membership, reliability)
        subtraction = (
            torch.sigmoid(self.artifact_lambda)[None, None, :, None]
            * (1.0 - region_reliability[:, None, :, None])
            * artifact_region
        )
        region_features = self.region_norm(signal_region - subtraction)
        region_features = (
            region_features * self.personal_region_scale[None, None, :, None]
        )

        adjacency = self._adjacency(edge_mask)
        support0 = region_features
        support1 = torch.einsum("rs,btsd->btrd", adjacency, support0)
        support2 = 2.0 * torch.einsum(
            "rs,btsd->btrd", adjacency, support1
        ) - support0
        projected = torch.stack(
            [
                layer(value)
                for layer, value in zip(
                    self.order_projections, (support0, support1, support2)
                )
            ],
            dim=1,
        )
        order_weights = torch.softmax(
            self.order_gate(region_features.mean(dim=(1, 2))), dim=-1
        )
        graph_features = (
            projected * order_weights[:, :, None, None, None]
        ).sum(dim=1)

        channel_weight = reliability / reliability.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-6)
        glance = torch.einsum(
            "bc,btcd->btd", channel_weight, channel_features
        )
        focus_gate, focus_scores = self._focus_gate(graph_features)
        focus = torch.einsum("btr,btrd->btd", focus_gate, graph_features)
        graph_pool = graph_features.mean(dim=2)
        reliability_summary = torch.stack(
            (
                reliability.mean(dim=1), reliability.std(dim=1),
                region_reliability.mean(dim=1),
                region_reliability.std(dim=1),
            ),
            dim=-1,
        )
        fusion_branches = torch.stack(
            (
                self.glance_projection(glance),
                self.focus_projection(focus),
                self.graph_projection(graph_pool),
            ),
            dim=2,
        )
        if self.fusion_mode == "reliability":
            summary = reliability_summary[:, None].expand(-1, time, -1)
            fusion_logits = self.fusion_gate(torch.cat(
                (glance, focus, graph_pool, summary), dim=-1
            ))
            fusion_weights_time = torch.softmax(fusion_logits, dim=-1)
            fused_time = (
                fusion_weights_time[:, :, :, None] * fusion_branches
            ).sum(dim=2)
        else:
            fusion_weights_time = fusion_branches.new_full(
                (batch, time, 3), 1.0 / 3.0
            )
            fused_time = fusion_branches.sum(dim=2)
        temporal_input = fused_time.transpose(1, 2)
        fused_time = fused_time + self.temporal_output_mix(
            F.gelu(self.temporal_output_depthwise(temporal_input))
        ).transpose(1, 2)
        temporal_weights = torch.softmax(
            self.temporal_output_score(fused_time).squeeze(-1), dim=-1
        )
        embedding = self.embedding_norm(torch.einsum(
            "bt,btd->bd", temporal_weights, fused_time
        ))
        embedding = self.embedding_dropout(embedding)
        embedding = embedding * self.personal_scale + self.personal_bias
        glance_pool = torch.einsum("bt,btd->bd", temporal_weights, glance)
        outputs = {
            "logits": self.head(embedding),
            "coarse_logits": self.coarse_head(glance_pool),
            "embedding": embedding,
            "glance": glance_pool,
            "graph_pool": torch.einsum(
                "bt,btd->bd", temporal_weights, graph_pool
            ),
            "reliability": reliability,
            "region_reliability": region_reliability,
            "signal_attention": signal_weights.mean(dim=1),
            "artifact_attention": artifact_weights.mean(dim=1),
            "adjacency": adjacency,
            "order_weights": order_weights,
            "focus_gate": focus_gate.mean(dim=1),
            "focus_scores": focus_scores.mean(dim=1),
            "fusion_weights": fusion_weights_time.mean(dim=1),
            "temporal_weights": temporal_weights,
        }
        stable_summary = torch.cat(
            (sequence.mean(dim=2), sequence.std(dim=2)), dim=-1
        ).reshape(batch, -1)
        outputs["stable_logits"] = self.stable_head(stable_summary)
        outputs["stable_fusion_weight"] = torch.sigmoid(
            self.stable_fusion_logit
        ).expand(batch)
        if self.structured_evidence:
            evidence = self.structured_evidence_features(
                inputs if evidence_inputs is None else evidence_inputs
            )
            evidence = (
                evidence - self.evidence_feature_mean[None]
            ) / self.evidence_feature_scale[None]
            split = self.channels * 10
            outputs["stable_logits"] = self.stable_head(evidence[:, :split])
            outputs["connection_logits"] = self.connection_evidence_head(
                evidence[:, split:]
            )
            quality = reliability.mean(dim=1).clamp(0.0, 1.0)
            graph_weight = 0.01 * torch.sigmoid(
                self.decision_fusion_logits
            ) * quality
            outputs["decision_fusion_weights"] = torch.stack(
                (torch.ones_like(quality), torch.ones_like(quality), graph_weight),
                dim=1,
            )
        if self.classes == 4:
            outputs["valence_logits"] = self.valence_head(embedding)
            outputs["arousal_logits"] = self.arousal_head(embedding)
        return outputs


    def combined_logits(
        self,
        outputs: dict[str, torch.Tensor],
        auxiliary_weight: float | None = None,
    ) -> torch.Tensor:
        logits = outputs["logits"]
        if "connection_logits" in outputs:
            graph_weight = outputs["decision_fusion_weights"][:, 2, None]
            logits = (
                outputs["stable_logits"]
                + outputs["connection_logits"]
                + graph_weight * torch.tanh(logits)
            )
        elif "stable_logits" in outputs:
            stable_weight = torch.sigmoid(self.stable_fusion_logit)
            logits = (
                (1.0 - stable_weight) * logits
                + stable_weight * outputs["stable_logits"]
            )
        if self.classes != 4:
            return logits
        if auxiliary_weight is None:
            auxiliary_weight = self.auxiliary_weight
        valence = F.log_softmax(outputs["valence_logits"], dim=-1)
        arousal = F.log_softmax(outputs["arousal_logits"], dim=-1)
        class_valence = torch.tensor((0, 1, 0, 1), device=logits.device)
        class_arousal = torch.tensor((0, 0, 1, 1), device=logits.device)
        auxiliary = valence[:, class_valence] + arousal[:, class_arousal]
        return logits + auxiliary_weight * auxiliary

    def forward(
        self,
        inputs: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[1] != self.channels:
            raise ValueError(f"expected B x {self.channels} x F, got {tuple(inputs.shape)}")
        adapted_inputs = inputs * self.personal_band_scale[None, None, :]
        if self.temporal_mode:
            return self._forward_temporal(
                adapted_inputs, edge_mask, evidence_inputs=inputs
            )
        channel_features = self._encode_channels(adapted_inputs)
        channel_features = (
            channel_features * self.personal_channel_scale[None, :, None]
        )
        reliability = self._reliability(adapted_inputs)
        scale = math.sqrt(self.hidden_dim)

        signal_logits = torch.einsum(
            "bcd,rd->brc", self.signal_key(channel_features), self.signal_query
        ) / scale
        signal_weights = self._masked_attention(signal_logits + reliability[:, None, :].log())
        artifact_logits = torch.einsum(
            "bcd,rd->brc", self.artifact_key(channel_features), self.artifact_query
        ) / scale
        artifact_logits = artifact_logits + 2.0 * (1.0 - reliability[:, None, :])
        artifact_weights = self._masked_attention(artifact_logits)
        signal_region = torch.einsum(
            "brc,bcd->brd", signal_weights, self.signal_value(channel_features)
        )
        artifact_region = torch.einsum(
            "brc,bcd->brd", artifact_weights, self.artifact_value(channel_features)
        )
        membership = self.membership / self.membership.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        region_reliability = torch.einsum("rc,bc->br", membership, reliability)
        subtraction = (
            torch.sigmoid(self.artifact_lambda)[None, :, None]
            * (1.0 - region_reliability[:, :, None])
            * artifact_region
        )
        region_features = self.region_norm(signal_region - subtraction)
        region_features = (
            region_features * self.personal_region_scale[None, :, None]
        )

        adjacency = self._adjacency(edge_mask)
        support0 = region_features
        support1 = torch.einsum("rs,bsd->brd", adjacency, support0)
        support2 = 2.0 * torch.einsum("rs,bsd->brd", adjacency, support1) - support0
        projected = torch.stack(
            [layer(value) for layer, value in zip(self.order_projections, (support0, support1, support2))],
            dim=1,
        )
        order_weights = torch.softmax(self.order_gate(region_features.mean(dim=1)), dim=-1)
        graph_features = (projected * order_weights[:, :, None, None]).sum(dim=1)

        channel_weight = reliability / reliability.sum(dim=1, keepdim=True).clamp_min(1e-6)
        glance = torch.einsum("bc,bcd->bd", channel_weight, channel_features)
        focus_gate, focus_scores = self._focus_gate(graph_features)
        focus = torch.einsum("br,brd->bd", focus_gate, graph_features)
        graph_pool = graph_features.mean(dim=1)
        reliability_summary = torch.stack(
            (
                reliability.mean(dim=1), reliability.std(dim=1),
                region_reliability.mean(dim=1),
                region_reliability.std(dim=1),
            ),
            dim=-1,
        )
        fusion_branches = torch.stack(
            (
                self.glance_projection(glance),
                self.focus_projection(focus),
                self.graph_projection(graph_pool),
            ),
            dim=1,
        )
        if self.fusion_mode == "reliability":
            fusion_logits = self.fusion_gate(torch.cat(
                (glance, focus, graph_pool, reliability_summary), dim=-1
            ))
            fusion_weights = torch.softmax(fusion_logits, dim=-1)
            fused = (fusion_weights[:, :, None] * fusion_branches).sum(dim=1)
        else:
            # Backward-compatible ablation matching the original unweighted
            # sum. LayerNorm makes sum and mean equivalent up to epsilon.
            fusion_weights = fusion_branches.new_full(
                (len(inputs), 3), 1.0 / 3.0
            )
            fused = fusion_branches.sum(dim=1)
        embedding = self.embedding_norm(fused)
        embedding = self.embedding_dropout(embedding)
        embedding = embedding * self.personal_scale + self.personal_bias
        outputs = {
            "logits": self.head(embedding),
            "coarse_logits": self.coarse_head(glance),
            "embedding": embedding,
            "glance": glance,
            "graph_pool": graph_pool,
            "reliability": reliability,
            "region_reliability": region_reliability,
            "signal_attention": signal_weights,
            "artifact_attention": artifact_weights,
            "adjacency": adjacency,
            "order_weights": order_weights,
            "focus_gate": focus_gate,
            "focus_scores": focus_scores,
            "fusion_weights": fusion_weights,
        }
        if self.classes == 4:
            outputs["valence_logits"] = self.valence_head(embedding)
            outputs["arousal_logits"] = self.arousal_head(embedding)
        return outputs
