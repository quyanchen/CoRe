from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, GCNConv


def segment_sum(x: torch.Tensor, segment: torch.Tensor, num_segments: int) -> torch.Tensor:
    out = x.new_zeros((num_segments, x.size(-1)))
    out.index_add_(0, segment, x)
    return out


def segment_mean(x: torch.Tensor, segment: torch.Tensor, num_segments: int) -> torch.Tensor:
    out = segment_sum(x, segment, num_segments)
    counts = x.new_zeros((num_segments, 1))
    counts.index_add_(0, segment, torch.ones((x.size(0), 1), device=x.device, dtype=x.dtype))
    return out / counts.clamp_min(1.0)


def pad_by_graph(
    node_h: torch.Tensor,
    graph_id: torch.Tensor,
    num_graphs: int,
    keep_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    hidden_dim = node_h.size(-1)
    indices = [torch.nonzero(graph_id == gid, as_tuple=False).flatten() for gid in range(num_graphs)]
    max_nodes = max(int(idx.numel()) for idx in indices)
    padded = node_h.new_zeros((num_graphs, max_nodes, hidden_dim))
    key_padding_mask = torch.ones((num_graphs, max_nodes), device=node_h.device, dtype=torch.bool)
    for gid, idx in enumerate(indices):
        n = int(idx.numel())
        padded[gid, :n] = node_h[idx]
        if keep_mask is None:
            key_padding_mask[gid, :n] = False
        else:
            local_keep = keep_mask[idx].bool()
            if not torch.any(local_keep):
                local_keep[0] = True
            key_padding_mask[gid, :n] = ~local_keep
    return padded, key_padding_mask, indices


def unpad_by_graph(padded: torch.Tensor, indices: Sequence[torch.Tensor], total_nodes: int) -> torch.Tensor:
    node_h = padded.new_zeros((total_nodes, padded.size(-1)))
    for gid, idx in enumerate(indices):
        node_h[idx] = padded[gid, : idx.numel()]
    return node_h


class HIGConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.intra_conv = GCNConv(in_dim, out_dim)
        self.cross_conv = GATConv(in_dim, out_dim // num_heads, heads=num_heads, dropout=dropout, add_self_loops=False)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(h)
        intra_mask = edge_type == 0
        cross_mask = edge_type == 1
        if torch.any(intra_mask):
            out = out + self.intra_conv(h, edge_index[:, intra_mask])
        if torch.any(cross_mask):
            out = out + self.cross_conv(h, edge_index[:, cross_mask])
        return self.norm(h + self.dropout(F.gelu(out)))


class HIGEncoder(nn.Module):
    def __init__(self, in_dim: int = 2048, hidden_dim: int = 256, num_layers: int = 3, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.layers = nn.ModuleList([HIGConv(hidden_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, graph: Batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.input_proj(graph.x.float())
        edge_index = graph.edge_index.to(h.device)
        edge_type = graph.edge_type.to(h.device)
        for layer in self.layers:
            h = layer(h, edge_index, edge_type)

        side = graph.side.to(h.device).long()
        graph_id = graph.batch.to(h.device).long()
        num_graphs = int(graph.num_graphs)
        base_a = segment_mean(h[side == 0], graph_id[side == 0], num_graphs)
        base_b = segment_mean(h[side == 1], graph_id[side == 1], num_graphs)
        return h, base_a, base_b, graph_id


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedReciprocalMotifBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.summary_to_motif = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.motif_to_summary = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.motif_self = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.summary_self = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.motif_norm1 = nn.LayerNorm(hidden_dim)
        self.motif_norm2 = nn.LayerNorm(hidden_dim)
        self.motif_norm3 = nn.LayerNorm(hidden_dim)
        self.summary_norm1 = nn.LayerNorm(hidden_dim)
        self.summary_norm2 = nn.LayerNorm(hidden_dim)
        self.summary_norm3 = nn.LayerNorm(hidden_dim)
        self.motif_ffn = FeedForward(hidden_dim, dropout)
        self.summary_ffn = FeedForward(hidden_dim, dropout)

    def forward(
        self,
        node_h: torch.Tensor,
        summary_tokens: torch.Tensor,
        graph_id: torch.Tensor,
        node_keep_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_graphs = summary_tokens.size(0)
        motif_tokens, key_padding_mask, indices = pad_by_graph(node_h, graph_id, num_graphs, node_keep_mask)

        motif_update, _ = self.motif_to_summary(motif_tokens, summary_tokens, summary_tokens)
        motif_tokens = self.motif_norm1(motif_tokens + motif_update)
        motif_self, _ = self.motif_self(motif_tokens, motif_tokens, motif_tokens, key_padding_mask=key_padding_mask)
        motif_tokens = self.motif_norm2(motif_tokens + motif_self)
        motif_tokens = self.motif_norm3(motif_tokens + self.motif_ffn(motif_tokens))

        summary_update, _ = self.summary_to_motif(summary_tokens, motif_tokens, motif_tokens, key_padding_mask=key_padding_mask)
        summary_tokens = self.summary_norm1(summary_tokens + summary_update)
        summary_self, _ = self.summary_self(summary_tokens, summary_tokens, summary_tokens)
        summary_tokens = self.summary_norm2(summary_tokens + summary_self)
        summary_tokens = self.summary_norm3(summary_tokens + self.summary_ffn(summary_tokens))

        return unpad_by_graph(motif_tokens, indices, node_h.size(0)), summary_tokens


class TripletRoleDDIModel(nn.Module):
    def __init__(
        self,
        num_relations: int,
        frag_dim: int = 2048,
        hidden_dim: int = 256,
        hig_layers: int = 3,
        role_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        fusion_dim: int = 8,
    ):
        super().__init__()
        if fusion_dim not in {3, 5, 7, 8}:
            raise ValueError(f"fusion_dim must be one of 3, 5, 7, or 8, got {fusion_dim}")
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.fusion_dim = fusion_dim
        self.encoder = HIGEncoder(frag_dim, hidden_dim, hig_layers, num_heads, dropout)
        self.rel_emb = nn.Embedding(num_relations, hidden_dim)
        self.summary_init = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.summary_blocks = nn.ModuleList(
            [SharedReciprocalMotifBlock(hidden_dim, num_heads, dropout) for _ in range(role_layers)]
        )

        self.evidence_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.role_query_a = nn.Linear(hidden_dim * 3, hidden_dim)
        self.role_query_b = nn.Linear(hidden_dim * 3, hidden_dim)
        self.triplet_query = nn.Linear(hidden_dim * 5, hidden_dim)
        self.shift_a = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.shift_b = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim))
        self.shift_gate = nn.Linear(hidden_dim, hidden_dim)
        self.role_norm = nn.LayerNorm(hidden_dim)
        self.triplet_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * fusion_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _encode_pair(self, graph: Batch, node_keep_mask: Optional[torch.Tensor] = None):
        node_h, base_a, base_b, graph_id = self.encoder(graph)
        side = graph.side.to(node_h.device).long()
        summary = self.summary_init(torch.cat([base_a, base_b, torch.abs(base_a - base_b)], dim=-1))
        summary_tokens = torch.stack([summary, base_a, base_b], dim=1)
        for block in self.summary_blocks:
            node_h, summary_tokens = block(node_h, summary_tokens, graph_id, node_keep_mask)
        pair_summary, refined_a, refined_b = summary_tokens[:, 0], summary_tokens[:, 1], summary_tokens[:, 2]
        base_a = base_a + refined_a
        base_b = base_b + refined_b
        return node_h, base_a, base_b, pair_summary, graph_id, side

    def _relation_readout(
        self,
        node_h: torch.Tensor,
        base_a: torch.Tensor,
        base_b: torch.Tensor,
        pair_summary: torch.Tensor,
        graph_id: torch.Tensor,
        side: torch.Tensor,
        evidence_keep_mask: Optional[torch.Tensor] = None,
    ):
        batch_size = base_a.size(0)
        rel = self.rel_emb.weight.unsqueeze(0).expand(batch_size, -1, -1)
        base_a_r = base_a.unsqueeze(1).expand(-1, self.num_relations, -1)
        base_b_r = base_b.unsqueeze(1).expand(-1, self.num_relations, -1)
        pair_r = pair_summary.unsqueeze(1).expand(-1, self.num_relations, -1)

        triplet_q = self.triplet_query(torch.cat([base_a_r, rel, base_b_r, base_a_r * base_b_r, torch.abs(base_a_r - base_b_r)], dim=-1))
        query_a = self.role_query_a(torch.cat([base_a_r, rel, pair_r], dim=-1))
        query_b = self.role_query_b(torch.cat([base_b_r, rel, pair_r], dim=-1))

        evidence_a, attn_a = self._side_relation_evidence(
            node_h,
            query_a,
            graph_id,
            side,
            side_value=0,
            evidence_keep_mask=evidence_keep_mask,
        )
        evidence_b, attn_b = self._side_relation_evidence(
            node_h,
            query_b,
            graph_id,
            side,
            side_value=1,
            evidence_keep_mask=evidence_keep_mask,
        )

        delta_a = self.shift_a(torch.cat([query_a, triplet_q, evidence_a, query_b], dim=-1))
        delta_b = self.shift_b(torch.cat([query_b, triplet_q, evidence_b, query_a], dim=-1))
        gate_a = torch.sigmoid(self.shift_gate(evidence_a))
        gate_b = torch.sigmoid(self.shift_gate(evidence_b))
        role_a = self.role_norm(base_a_r + gate_a * delta_a)
        role_b = self.role_norm(base_b_r + gate_b * delta_b)
        triplet = self.triplet_norm(triplet_q + pair_r + evidence_a + evidence_b)

        shift_a = role_a - base_a_r
        shift_b = role_b - base_b_r
        feature_parts = [triplet, role_a, role_b]
        if self.fusion_dim >= 7:
            feature_parts.extend([shift_a, shift_b])
        if self.fusion_dim >= 5:
            feature_parts.extend([role_a * role_b, torch.abs(role_a - role_b)])
        if self.fusion_dim == 8:
            feature_parts.append(rel)
        features = torch.cat(feature_parts, dim=-1)
        logits = self.classifier(features).squeeze(-1)
        node_attention = torch.zeros((node_h.size(0), self.num_relations), device=node_h.device, dtype=node_h.dtype)
        node_attention[side == 0] = attn_a[side == 0]
        node_attention[side == 1] = attn_b[side == 1]
        return {
            "logits": logits,
            "triplet_all": triplet,
            "role_a_all": role_a,
            "role_b_all": role_b,
            "shift_a_all": shift_a,
            "shift_b_all": shift_b,
            "node_attention_all": node_attention,
        }

    def _side_relation_evidence(
        self,
        node_h: torch.Tensor,
        query: torch.Tensor,
        graph_id: torch.Tensor,
        side: torch.Tensor,
        side_value: int,
        evidence_keep_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_relations, hidden_dim = query.shape
        key = self.evidence_key(node_h)
        scores = (key.unsqueeze(1) * query[graph_id]).sum(dim=-1) / (hidden_dim ** 0.5)

        side_valid = (side == side_value).unsqueeze(1).expand(-1, num_relations)
        if evidence_keep_mask is None:
            valid = side_valid
        else:
            if evidence_keep_mask.shape != scores.shape:
                raise ValueError(
                    "evidence_keep_mask must have shape "
                    f"{tuple(scores.shape)}, got {tuple(evidence_keep_mask.shape)}"
                )
            valid = side_valid & evidence_keep_mask.to(device=scores.device, dtype=torch.bool)

        attn = torch.zeros_like(scores)
        min_value = torch.finfo(scores.dtype).min
        for gid in range(batch_size):
            idx = torch.nonzero((graph_id == gid) & (side == side_value), as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            local_valid = valid[idx]
            missing_relations = ~local_valid.any(dim=0)
            if torch.any(missing_relations):
                missing = torch.nonzero(missing_relations, as_tuple=False).flatten().tolist()
                raise ValueError(
                    f"Evidence mask removes every motif for graph {gid}, "
                    f"side {side_value}, relations {missing}."
                )
            local_scores = scores[idx].masked_fill(~local_valid, min_value)
            attn[idx] = torch.softmax(local_scores, dim=0)

        evidence = torch.zeros(
            (batch_size, num_relations, hidden_dim),
            device=node_h.device,
            dtype=node_h.dtype,
        )
        evidence.index_add_(0, graph_id, node_h.unsqueeze(1) * attn.unsqueeze(-1))
        return evidence, attn

    def _gather_relation_outputs(self, outputs: Dict[str, torch.Tensor], relation: torch.Tensor, graph_id: torch.Tensor):
        idx = relation.to(outputs["logits"].device).view(-1, 1, 1)
        idx_h = idx.expand(-1, 1, self.hidden_dim)
        node_relation = relation.to(outputs["logits"].device)[graph_id].view(-1, 1)
        return {
            "triplet": outputs["triplet_all"].gather(1, idx_h).squeeze(1),
            "role_a": outputs["role_a_all"].gather(1, idx_h).squeeze(1),
            "role_b": outputs["role_b_all"].gather(1, idx_h).squeeze(1),
            "shift_a": outputs["shift_a_all"].gather(1, idx_h).squeeze(1),
            "shift_b": outputs["shift_b_all"].gather(1, idx_h).squeeze(1),
            "node_attention": outputs["node_attention_all"].gather(1, node_relation).squeeze(1),
        }

    def forward(
        self,
        graph: Batch,
        relation: Optional[torch.Tensor] = None,
        node_keep_mask: Optional[torch.Tensor] = None,
        evidence_keep_mask: Optional[torch.Tensor] = None,
    ):
        node_h, base_a, base_b, pair_summary, graph_id, side = self._encode_pair(graph, node_keep_mask)
        outputs = self._relation_readout(
            node_h,
            base_a,
            base_b,
            pair_summary,
            graph_id,
            side,
            evidence_keep_mask=evidence_keep_mask,
        )
        if relation is None:
            relation = outputs["logits"].argmax(dim=-1)
        gathered = self._gather_relation_outputs(outputs, relation, graph_id)
        gathered.update(
            {
                "logits": outputs["logits"],
                "base_a": base_a,
                "base_b": base_b,
                "node_attention_all": outputs["node_attention_all"],
                "graph_id": graph_id,
                "side": side,
                "node_h": node_h,
            }
        )
        return gathered


def weighted_triplet_role_alignment_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    drug_a: Sequence[str],
    drug_b: Sequence[str],
    temperature: float = 0.1,
    partner_weight: float = 2.0,
    relation_weight: float = 2.0,
    direction_weight: float = 2.0,
) -> torch.Tensor:
    triplet = F.normalize(outputs["triplet"], dim=-1)
    role_pair = F.normalize(outputs["role_a"] + outputs["role_b"], dim=-1)
    sim = triplet @ role_pair.t() / temperature
    n = sim.size(0)
    device = sim.device
    weights = torch.ones((n, n), device=device)
    labels_cpu = labels.detach().cpu().tolist()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if drug_a[i] == drug_a[j] or drug_a[i] == drug_b[j] or drug_b[i] == drug_a[j] or drug_b[i] == drug_b[j]:
                weights[i, j] += partner_weight
            if labels_cpu[i] == labels_cpu[j]:
                weights[i, j] += relation_weight
            if drug_a[i] == drug_b[j] and drug_b[i] == drug_a[j]:
                weights[i, j] += direction_weight
    weights.fill_diagonal_(1.0)
    exp_sim = torch.exp(sim - sim.max(dim=1, keepdim=True).values.detach())
    pos = exp_sim.diag()
    denom = (exp_sim * weights).sum(dim=1).clamp_min(1e-12)
    forward = -torch.log(pos / denom).mean()
    exp_sim_t = exp_sim.t()
    denom_t = (exp_sim_t * weights.t()).sum(dim=1).clamp_min(1e-12)
    backward = -torch.log(exp_sim_t.diag() / denom_t).mean()
    return 0.5 * (forward + backward)


def base_role_regularization(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    shift_a = outputs["shift_a"]
    shift_b = outputs["shift_b"]
    base_a = F.normalize(outputs["base_a"], dim=-1)
    base_b = F.normalize(outputs["base_b"], dim=-1)
    role_a = F.normalize(outputs["role_a"], dim=-1)
    role_b = F.normalize(outputs["role_b"], dim=-1)
    bounded_shift = 0.5 * (shift_a.pow(2).mean() + shift_b.pow(2).mean())
    chemical_grounding = 0.5 * ((1.0 - (base_a * role_a).sum(dim=-1)).mean() + (1.0 - (base_b * role_b).sum(dim=-1)).mean())
    return bounded_shift + chemical_grounding


def context_role_separation_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    drug_a: Sequence[str],
    drug_b: Sequence[str],
    margin: float = 0.45,
) -> torch.Tensor:
    roles: List[torch.Tensor] = []
    drug_ids: List[str] = []
    contexts: List[Tuple[int, str, str]] = []
    labels_cpu = labels.detach().cpu().tolist()
    for i, label in enumerate(labels_cpu):
        roles.append(outputs["role_a"][i])
        drug_ids.append(drug_a[i])
        contexts.append((label, drug_b[i], "head"))
        roles.append(outputs["role_b"][i])
        drug_ids.append(drug_b[i])
        contexts.append((label, drug_a[i], "tail"))
    if len(roles) < 2:
        return outputs["logits"].new_tensor(0.0)
    role_tensor = F.normalize(torch.stack(roles, dim=0), dim=-1)
    penalties = []
    for i in range(len(roles)):
        for j in range(i + 1, len(roles)):
            if drug_ids[i] == drug_ids[j] and contexts[i] != contexts[j]:
                penalties.append(F.relu((role_tensor[i] * role_tensor[j]).sum() - margin))
    if not penalties:
        return outputs["logits"].new_tensor(0.0)
    return torch.stack(penalties).mean()


def build_counterfactual_masks(
    node_attention: torch.Tensor,
    graph_id: torch.Tensor,
    side: Optional[torch.Tensor] = None,
    top_ratio: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = node_attention.device
    top_keep = torch.ones_like(node_attention, dtype=torch.bool, device=device)
    random_keep = torch.ones_like(node_attention, dtype=torch.bool, device=device)
    low_keep = torch.ones_like(node_attention, dtype=torch.bool, device=device)
    num_graphs = int(graph_id.max().item()) + 1
    for gid in range(num_graphs):
        side_values = [0, 1] if side is not None else [None]
        for side_value in side_values:
            idx = torch.nonzero(graph_id == gid, as_tuple=False).flatten()
            if side_value is not None:
                idx = idx[side[idx] == side_value]
            if idx.numel() <= 1:
                continue
            k = min(idx.numel() - 1, max(1, int(idx.numel() * top_ratio)))
            local_scores = node_attention[idx]
            top_idx = idx[torch.topk(local_scores, k=k, largest=True).indices]
            low_idx = idx[torch.topk(local_scores, k=k, largest=False).indices]
            rand_idx = idx[torch.randperm(idx.numel(), device=device)[:k]]
            top_keep[top_idx] = False
            random_keep[rand_idx] = False
            low_keep[low_idx] = False
    return top_keep, random_keep, low_keep


def counterfactual_motif_fidelity_loss(
    logits: torch.Tensor,
    top_logits: torch.Tensor,
    random_logits: torch.Tensor,
    low_logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.15,
    random_tolerance: float = 0.05,
) -> torch.Tensor:
    prob = F.softmax(logits, dim=-1).gather(1, labels.view(-1, 1)).squeeze(1)
    top_prob = F.softmax(top_logits, dim=-1).gather(1, labels.view(-1, 1)).squeeze(1)
    random_prob = F.softmax(random_logits, dim=-1).gather(1, labels.view(-1, 1)).squeeze(1)
    low_prob = F.softmax(low_logits, dim=-1).gather(1, labels.view(-1, 1)).squeeze(1)
    top_drop = prob.detach() - top_prob
    random_drop = (prob.detach() - random_prob).abs()
    low_drop = (prob.detach() - low_prob).abs()
    faithful_drop = F.relu(margin - top_drop).mean()
    random_stability = F.relu(random_drop - random_tolerance).mean()
    low_stability = F.relu(low_drop - random_tolerance).mean()
    ranking = F.relu(margin - (top_drop - low_drop.detach())).mean()
    return faithful_drop + random_stability + low_stability + ranking
