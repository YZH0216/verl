# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FSDP sharded checkpoints keep replicated entries (buffers, and any parameter
excluded from wrapping) as plain tensors rather than DTensors: every rank stores
an identical full copy. The merger must de-duplicate them, not concatenate them.
"""

import torch

from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger

WORLD_SIZE = 4


def _write_shards(tmp_path, per_rank_state_dict):
    for rank in range(WORLD_SIZE):
        torch.save(
            per_rank_state_dict(rank),
            tmp_path / f"model_world_size_{WORLD_SIZE}_rank_{rank}.pt",
        )


def _merge(tmp_path):
    merger = FSDPModelMerger.__new__(FSDPModelMerger)
    merger.config = ModelMergerConfig(operation="merge", backend="fsdp", local_dir=str(tmp_path))
    return merger._load_and_merge_state_dicts(
        world_size=WORLD_SIZE, total_shards=WORLD_SIZE, mesh_shape=(WORLD_SIZE,), mesh_dim_names=("fsdp",)
    )


def test_replicated_entries_are_deduplicated(tmp_path):
    """A replicated buffer must keep its original shape after merging."""
    scalar = torch.tensor([0.5], dtype=torch.bfloat16)
    vector = torch.arange(8, dtype=torch.bfloat16)

    _write_shards(tmp_path, lambda rank: {"layer_scalar": scalar.clone(), "std_scale": vector.clone()})

    merged = _merge(tmp_path)

    assert merged["layer_scalar"].shape == scalar.shape
    assert merged["std_scale"].shape == vector.shape
    torch.testing.assert_close(merged["layer_scalar"], scalar)
    torch.testing.assert_close(merged["std_scale"], vector)


def test_mismatched_replicated_shapes_are_rejected(tmp_path):
    """Shapes differing across ranks mean the entry was not replicated: fail loudly."""
    _write_shards(tmp_path, lambda rank: {"suspicious": torch.zeros(rank + 1, dtype=torch.bfloat16)})

    try:
        _merge(tmp_path)
    except AssertionError as e:
        assert "suspicious" in str(e)
    else:
        raise AssertionError("expected an AssertionError for shapes that differ across ranks")
