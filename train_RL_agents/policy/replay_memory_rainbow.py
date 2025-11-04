import torch
from collections import deque, defaultdict
from tensordict import TensorDict
from torchrl.data import ReplayBuffer as TorchRLReplayBuffer
from torchrl.data import TensorStorage, PrioritizedSampler


class ReplayMemory:
    """
    TorchRL-based Prioritized Experience Replay for Rainbow with n-step return support.
    - Uses TensorStorage on CUDA and PrioritizedSampler for sampling.
    - Computes n-step returns online via per-stream deques to avoid lookahead at sample time.
    - Exposes a backward-compatible API used by Agent.train_Rainbow():
        sample(B) -> (indices, states, actions, returns, next_states, nonterminals, weights)
        update_priorities(indices, priorities)
    """

    def __init__(self, device, capacity, gamma=0.99, n=3, priority_exponent=0.5, priority_weight=0.4):
        self.device = torch.device(device)
        self.capacity = int(capacity)
        self.gamma = float(gamma)
        self.n = int(n)
        self.priority_exponent = float(priority_exponent)
        self.priority_weight = float(priority_weight)

        self._rb = TorchRLReplayBuffer(
            storage=TensorStorage(max_size=self.capacity, device=self.device),
            sampler=PrioritizedSampler(),
        )

        # Online n-step buffers keyed by a unique stream id (e.g., env_id * R + robot_id)
        self._streams = defaultdict(deque)
        self._gamma_vec = torch.tensor([self.gamma ** i for i in range(self.n)], dtype=torch.float32, device=self.device)

    # ---- Public API for trainer to append transitions ----
    def add_batch(self, td_batch: TensorDict, stream_ids: torch.Tensor):
        """
        Add a batch of 1-step transitions and internally commit matured n-step transitions to replay.
        td_batch keys expected on device:
          - self_state [B, 7]
          - objects_state [B, M, 5]
          - objects_mask [B, M]
          - action [B, 1] (int64) for discrete Rainbow
          - reward [B, 1]
          - done [B, 1] (bool)
          - next/self_state, next/objects_state, next/objects_mask
        stream_ids: Long tensor [B] unique per (env, robot) stream.
        """
        B = td_batch.shape[0]
        if not isinstance(stream_ids, torch.Tensor):
            raise TypeError("stream_ids must be a torch.Tensor")
        sid = stream_ids.to(torch.long, device=torch.device("cpu"))  # use CPU indices for dict keys

        # Iterate over batch items to update streams; mature transitions are pushed to RB
        for i in range(B):
            key = int(sid[i].item())
            step = {
                "self_state": td_batch.get("self_state")[i],
                "objects_state": td_batch.get("objects_state")[i],
                "objects_mask": td_batch.get("objects_mask")[i],
                "action": td_batch.get("action")[i].to(dtype=torch.int64).view(1),
                "reward": td_batch.get("reward")[i].to(dtype=torch.float32).view(1),
                "done": td_batch.get("done")[i].view(1),
                "next_self_state": td_batch.get(("next", "self_state"))[i],
                "next_objects_state": td_batch.get(("next", "objects_state"))[i],
                "next_objects_mask": td_batch.get(("next", "objects_mask"))[i],
            }
            self._push_stream_step(key, step)

    def _push_stream_step(self, key: int, step: dict):
        buf = self._streams[key]
        buf.append(step)

        # If episode ended at this step: flush all partial n-steps
        if bool(step["done"].item()):
            while len(buf) > 0:
                self._commit_n_step_from_buffer(key, buf)
            return

        # Else, if we have at least n transitions, commit one and slide window
        if len(buf) >= self.n:
            self._commit_n_step_from_buffer(key, buf)

    def _commit_n_step_from_buffer(self, key: int, buf: deque):
        # Create matured transition from the first up to n elements (or until done)
        n_avail = 0
        nonterminal = 1.0
        rewards = []
        for j, s in enumerate(buf):
            n_avail += 1
            rewards.append(s["reward"])  # [1]
            if bool(s["done"].item()):
                nonterminal = 0.0
                break
            if n_avail >= self.n:
                break

        # Compute discounted return
        rew = torch.cat(rewards, dim=0).to(self.device, dtype=torch.float32)  # [n_avail]
        gam = self._gamma_vec[: n_avail]
        Rn = torch.sum(rew * gam, dim=0, keepdim=False).view(1)  # [1]

        first = buf[0]
        last = buf[n_avail - 1]

        td = TensorDict(
            {
                "self_state": first["self_state"].to(self.device),
                "objects_state": first["objects_state"].to(self.device),
                "objects_mask": first["objects_mask"].to(self.device),
                "action": first["action"].to(self.device),  # [1]
                "returns": Rn.to(self.device),  # [1]
                "nonterminal": torch.tensor([[nonterminal]], dtype=torch.float32, device=self.device),
                ("next", "self_state"): last["next_self_state"].to(self.device),
                ("next", "objects_state"): last["next_objects_state"].to(self.device),
                ("next", "objects_mask"): last["next_objects_mask"].to(self.device),
            },
            batch_size=[1],
            device=self.device,
        )
        # Extend to RB; Priorities will be set/updated via update_priorities later
        self._rb.extend(td)
        # Slide window by one (standard n-step)
        buf.popleft()

    # ---- Sampling API compatible with Agent.train_Rainbow ----
    def sample(self, batch_size):
        td, info = self._rb.sample(batch_size, return_info=True)
        td = td.to(self.device)
        # Indices for priority update
        if isinstance(info, dict):
            idxs = info.get("index", None)
            probs = info.get("probability", None)
        else:
            idxs = None
            probs = None
        if idxs is None:
            # Fallback: fabricate indices in range
            B = td.batch_size[0]
            idxs = torch.arange(B, device=self.device, dtype=torch.long)
        # Importance weights
        if probs is None:
            # Approximate uniform probs
            B = td.batch_size[0]
            probs = torch.full((B,), 1.0 / max(1, len(self._rb._storage)), device=self.device)
        capacity = max(1, len(self._rb._storage))
        weights = (capacity * probs) ** -self.priority_weight
        weights = (weights / weights.max()).to(self.device, dtype=torch.float32)

        # Repackage as tuple expected by Agent
        states = (
            td.get("self_state"),
            td.get("objects_state"),
            td.get("objects_mask"),
        )
        next_states = (
            td.get(("next", "self_state")),
            td.get(("next", "objects_state")),
            td.get(("next", "objects_mask")),
        )
        actions = td.get("action").to(torch.int64).squeeze(-1)
        returns = td.get("returns").squeeze(-1)
        nonterminals = td.get("nonterminal")
        return idxs, states, actions, returns, next_states, nonterminals, weights

    def update_priorities(self, idxs, priorities):
        # PrioritizedSampler generally expects raw priorities; apply exponent here as in original
        prio = torch.as_tensor(priorities, dtype=torch.float32)
        prio = torch.pow(prio, self.priority_exponent).to(self.device)
        sampler = getattr(self._rb, "_sampler", None)
        if sampler is not None and hasattr(sampler, "update_priority"):
            sampler.update_priority(idxs.to(torch.long), prio)
        # If sampler API is different, fail silently (sampling will still work uniformly)

    # Compatibility helpers for trainer checks
    @property
    def transitions(self):
        class Dummy:
            def __init__(self, rb):
                self._rb = rb
            def num_elements(self):
                try:
                    return len(self._rb._storage)
                except Exception:
                    return len(self._rb)
        return Dummy(self._rb)
