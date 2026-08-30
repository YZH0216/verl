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

"""The dataset may select an agent loop per sample via an ``agent_name`` column;
``AgentLoopWorker.generate_sequences`` only fills in a default when that column is
absent. The fully-async generation path must not overwrite a caller-provided value.
"""

import numpy as np
import torch
from omegaconf import OmegaConf

from verl.experimental.fully_async_policy.detach_utils import prepare_single_generation_data


def _config(multi_turn_enable: bool):
    return OmegaConf.create({"actor_rollout_ref": {"rollout": {"n": 1, "multi_turn": {"enable": multi_turn_enable}}}})


def _batch_dict(agent_name=None):
    batch_dict = {"input_ids": torch.zeros(1, 4, dtype=torch.long)}
    if agent_name is not None:
        batch_dict["agent_name"] = np.array([agent_name], dtype=object)
    return batch_dict


def test_dataset_agent_name_is_preserved():
    out = prepare_single_generation_data(_batch_dict("my_custom_agent"), _config(multi_turn_enable=False))
    assert list(out.non_tensor_batch["agent_name"]) == ["my_custom_agent"]


def test_default_is_applied_when_absent():
    out = prepare_single_generation_data(_batch_dict(), _config(multi_turn_enable=False))
    assert list(out.non_tensor_batch["agent_name"]) == ["single_turn_agent"]


def test_multi_turn_path_leaves_column_untouched():
    out = prepare_single_generation_data(_batch_dict("my_custom_agent"), _config(multi_turn_enable=True))
    assert list(out.non_tensor_batch["agent_name"]) == ["my_custom_agent"]
