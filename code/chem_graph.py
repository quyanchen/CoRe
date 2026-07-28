from __future__ import annotations

import os
from itertools import combinations
from typing import Dict, Optional

import torch
from rdkit import Chem
from rdkit.Chem import AllChem, MolToSmiles, rdFingerprintGenerator, rdmolops
from torch_geometric.data import Data


def _safe_mol_from_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    return Chem.MolFromSmiles(Chem.MolToSmiles(mol))


def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> torch.Tensor:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return torch.zeros(n_bits, dtype=torch.float32)
    try:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp = generator.GetFingerprint(mol)
    except Exception:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return torch.tensor(list(fp), dtype=torch.float32)


def brics_fragment_graph(smiles: str, n_bits: int = 2048) -> Data:
    mol = _safe_mol_from_smiles(smiles)
    try:
        fragmented_mol = Chem.FragmentOnBRICSBonds(mol)
        atom_list = Chem.GetMolFrags(fragmented_mol, sanitizeFrags=True)
        fragment_mols = Chem.GetMolFrags(fragmented_mol, asMols=True, sanitizeFrags=True)
        fragments = [MolToSmiles(x, False) for x in fragment_mols]
    except Exception:
        atom_list = [tuple(range(mol.GetNumAtoms()))]
        fragments = [Chem.MolToSmiles(mol)]

    if not fragments:
        atom_list = [tuple(range(mol.GetNumAtoms()))]
        fragments = [Chem.MolToSmiles(mol)]

    adjacency_matrix = rdmolops.GetAdjacencyMatrix(mol)
    fragment_bonds = set()
    for frag1, frag2 in combinations(range(len(atom_list)), 2):
        for atom1 in atom_list[frag1]:
            for atom2 in atom_list[frag2]:
                if atom1 < mol.GetNumAtoms() and atom2 < mol.GetNumAtoms():
                    if adjacency_matrix[atom1, atom2] == 1 or atom1 == atom2:
                        fragment_bonds.add((frag1, frag2))
                        fragment_bonds.add((frag2, frag1))

    if fragment_bonds:
        src, dst = zip(*sorted(fragment_bonds))
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    x = torch.stack([morgan_fingerprint(f, n_bits=n_bits) for f in fragments])
    side = torch.zeros(len(fragments), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, side=side, num_nodes=len(fragments))
    data.fragment_smiles = fragments
    return data


def merge_drug_pair_graph(graph_a: Data, graph_b: Data) -> Data:
    n_a = int(graph_a.num_nodes)
    n_b = int(graph_b.num_nodes)

    edge_a = graph_a.edge_index
    edge_b = graph_b.edge_index + n_a
    intra_edge_index = torch.cat([edge_a, edge_b], dim=1)

    forward_src = torch.arange(n_a).repeat_interleave(n_b)
    forward_dst = torch.arange(n_b).repeat(n_a) + n_a
    backward_src = forward_dst
    backward_dst = forward_src
    cross_edge_index = torch.stack(
        [torch.cat([forward_src, backward_src]), torch.cat([forward_dst, backward_dst])],
        dim=0,
    )

    edge_index = torch.cat([intra_edge_index, cross_edge_index], dim=1)
    edge_type = torch.cat(
        [
            torch.zeros(intra_edge_index.size(1), dtype=torch.long),
            torch.ones(cross_edge_index.size(1), dtype=torch.long),
        ],
        dim=0,
    )
    x = torch.cat([graph_a.x, graph_b.x], dim=0)
    side = torch.cat([torch.zeros(n_a, dtype=torch.long), torch.ones(n_b, dtype=torch.long)], dim=0)
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type, side=side, num_nodes=n_a + n_b)
    data.fragment_smiles = list(getattr(graph_a, "fragment_smiles", [])) + list(getattr(graph_b, "fragment_smiles", []))
    data.num_frag_a = n_a
    data.num_frag_b = n_b
    return data


class FragmentGraphBank:
    def __init__(self, drug_to_smiles: Dict[str, str], n_bits: int = 2048, cache_path: Optional[str] = None):
        self.drug_to_smiles = drug_to_smiles
        self.n_bits = n_bits
        self.cache_path = cache_path
        self._cache: Dict[str, Data] = {}
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                if os.path.getsize(self.cache_path) > 0:
                    payload = torch.load(self.cache_path, map_location="cpu")
                    if payload.get("n_bits") == self.n_bits:
                        self._cache = payload.get("graphs", {})
            except (EOFError, RuntimeError, AttributeError, OSError):
                self._cache = {}

    def get(self, drug_id: str) -> Data:
        if drug_id not in self._cache:
            self._cache[drug_id] = brics_fragment_graph(self.drug_to_smiles[drug_id], n_bits=self.n_bits)
        return self._cache[drug_id]

    def precompute_all(self) -> None:
        missing = [drug_id for drug_id in self.drug_to_smiles if drug_id not in self._cache]
        for drug_id in missing:
            self._cache[drug_id] = brics_fragment_graph(self.drug_to_smiles[drug_id], n_bits=self.n_bits)
        self.save()

    def save(self) -> None:
        if not self.cache_path:
            return
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        tmp_path = self.cache_path + ".tmp"
        try:
            torch.save({"n_bits": self.n_bits, "graphs": self._cache}, tmp_path)
            os.replace(tmp_path, self.cache_path)
        except OSError:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def pair_graph(self, drug_a: str, drug_b: str) -> Data:
        return merge_drug_pair_graph(self.get(drug_a), self.get(drug_b))
