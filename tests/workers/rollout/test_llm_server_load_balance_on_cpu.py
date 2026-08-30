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

"""Least-loaded routing must not funnel every session onto one replica when the
inflight counts are tied, which is the normal state of an idle or lightly loaded
pool. The choice is pinned by the sticky-session cache for the rest of the
request, so a deterministic tie-break avalanches load onto a single replica.
"""

from verl.workers.rollout.llm_server import GlobalRequestLoadBalancer

NUM_SERVERS = 4
NUM_REQUESTS = 200


def _balancer(**kwargs):
    # Handles are irrelevant here: only the routing decision is under test.
    return GlobalRequestLoadBalancer(servers={f"server_{i}": object() for i in range(NUM_SERVERS)}, **kwargs)


def test_idle_pool_does_not_funnel_onto_one_replica():
    """Acquire+release repeatedly, so every decision is made with all counts at zero."""
    lb = _balancer()
    counts = dict.fromkeys(lb._inflight_requests, 0)

    for i in range(NUM_REQUESTS):
        server_id, _ = lb.acquire_server(f"request_{i}")
        counts[server_id] += 1
        lb.release_server(server_id)

    assert all(c > 0 for c in counts.values()), f"some replicas never received a session: {counts}"


def test_sticky_session_is_preserved():
    """Repeat turns of one request must keep landing on the same replica."""
    lb = _balancer()
    first, _ = lb.acquire_server("request_0")
    for _ in range(10):
        again, _ = lb.acquire_server("request_0")
        assert again == first


def test_least_loaded_still_wins_over_the_tie_break():
    """The tie-break only applies among equally-least-loaded replicas."""
    lb = _balancer()
    servers = list(lb._inflight_requests)
    for sid in servers[1:]:
        lb._inflight_requests[sid] = 5

    for i in range(20):
        server_id, _ = lb.acquire_server(f"request_{i}")
        assert server_id == servers[0]
        lb.release_server(server_id)


def test_full_determinism_routing_is_unchanged():
    """full_determinism=True must stay reproducible across balancer instances."""
    ids = [f"request_{i}" for i in range(NUM_REQUESTS)]
    first = [_balancer(full_determinism=True).acquire_server(r)[0] for r in ids]
    second = [_balancer(full_determinism=True).acquire_server(r)[0] for r in ids]
    assert first == second
