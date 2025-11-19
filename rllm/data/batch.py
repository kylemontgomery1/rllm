from dataclasses import dataclass, field

import numpy as np
import torch
from tensordict import TensorDict

__all__ = ["BatchProto"]


@dataclass
class BatchProto:
    batch: TensorDict = None
    non_tensor_batch: dict = field(default_factory=dict)
    meta_info: dict = field(default_factory=dict)

    def __post_init__(self):
        # Validate non_tensors share consistent leading dimension.
        if self.non_tensor_batch:
            if self.batch is not None:
                expected_len = int(self.batch.batch_size[0])
            else:
                any_key = next(iter(self.non_tensor_batch))
                expected_len = int(self.non_tensor_batch[any_key].shape[0])
            for name, arr in self.non_tensor_batch.items():
                assert isinstance(arr, np.ndarray), f"Non-tensor '{name}' must be a numpy ndarray"
                assert int(arr.shape[0]) == expected_len, f"Non-tensor '{name}' dim0={int(arr.shape[0])} != {expected_len}"

    def __len__(self) -> int:
        """Get the length of the batch."""
        if self.batch is not None:
            return int(self.batch.batch_size[0])
        if self.non_tensor_batch:
            key = next(iter(self.non_tensor_batch))
            return int(self.non_tensor_batch[key].shape[0])
        return 0

    def __getitem__(self, item) -> "BatchProto":
        """Get an item(s) from the batch."""
        if isinstance(item, int):
            tensor_data = self.batch[item] if self.batch is not None else None
            non_tensor_data = {key: val[item] for key, val in self.non_tensor_batch.items()}
            return type(self)(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=self.meta_info)
        if isinstance(item, slice):
            return self.slice(item.start, item.stop, item.step)
        elif isinstance(item, list | np.ndarray | torch.Tensor):
            return self.select(item)
        else:
            raise TypeError(f"Unsupported index type: {type(item)}")

    @classmethod
    def from_dict(
        cls,
        batch: dict[str, torch.Tensor] = None,
        non_tensor_batch: dict[str, np.ndarray] = None,
        meta_info: dict[str, np.ndarray] = None,
    ) -> "BatchProto":
        """Create a batch from a dictionary."""
        batch = batch or {}
        non_tensor_batch = non_tensor_batch or {}
        meta_info = meta_info or {}

        tensors = None
        if batch:
            expected_len = None
            for name, t in batch.items():
                assert isinstance(t, torch.Tensor), f"Tensor '{name}' must be a torch.Tensor"
                dim0 = int(t.shape[0]) if t.ndim > 0 else 1
                if expected_len is None:
                    expected_len = dim0
                else:
                    assert dim0 == expected_len, f"Tensor '{name}' dim0={dim0} != {expected_len}"
            tensors = TensorDict(source=batch, batch_size=(expected_len,))

        non_tensors = {}
        for k, v in non_tensor_batch.items():
            non_tensors[k] = v if isinstance(v, np.ndarray) else np.array(v, dtype=object)

        return cls(batch=tensors, non_tensor_batch=non_tensors, meta_info=dict(meta_info))

    def get(
        self,
        batch_keys: list[str] = None,
        non_tensor_batch_keys: list[str] = None,
        meta_info_keys: list[str] = None,
    ) -> "BatchProto":
        """Get a subset of the batch by keys."""
        batch_keys = batch_keys or []
        non_tensor_batch_keys = non_tensor_batch_keys or []
        meta_info_keys = meta_info_keys or []

        tensors = self.batch.select(*batch_keys) if (self.batch is not None and batch_keys) else self.batch
        non_tensors = {k: v for k, v in self.non_tensor_batch.items() if (not non_tensor_batch_keys or k in non_tensor_batch_keys)}
        meta = {k: v for k, v in self.meta_info.items() if (not meta_info_keys or k in meta_info_keys)}
        return BatchProto(batch=tensors, non_tensor_batch=non_tensors, meta_info=meta)

    def pop(
        self,
        batch_keys: list[str] = None,
        non_tensor_batch_keys: list[str] = None,
        meta_info_keys: list[str] = None,
    ) -> "BatchProto":
        """Pop a subset of the batch by keys."""
        batch_keys = batch_keys or []
        non_tensor_batch_keys = non_tensor_batch_keys or []
        meta_info_keys = meta_info_keys or []

        tensors = {}
        if self.batch is not None:
            for k in batch_keys:
                tensors[k] = self.batch.pop(k)
        non_tensors = {}
        for k in non_tensor_batch_keys:
            non_tensors[k] = self.non_tensor_batch.pop(k)
        meta = {}
        for k in meta_info_keys:
            meta[k] = self.meta_info.pop(k)

        tensors_td = TensorDict(source=tensors, batch_size=self.batch.batch_size if tensors else None) if tensors else None
        return BatchProto(batch=tensors_td, non_tensor_batch=non_tensors, meta_info=meta)

    def union(self, other: "BatchProto") -> "BatchProto":
        """Union two batches."""
        if self.batch is None:
            tensors = other.batch
        elif other.batch is None:
            tensors = self.batch
        else:
            tensors = self.batch.clone()
            for k, t in other.batch.items():
                if k in tensors.keys():
                    assert torch.equal(tensors[k], t), f"Tensor key '{k}' differs"
                else:
                    tensors.set(k, t)

        non_tensors = dict(self.non_tensor_batch)
        for k, v in other.non_tensor_batch.items():
            if k in non_tensors:
                a, b = non_tensors[k], v
                assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray), f"Non-tensor '{k}' must be ndarray"
                same = (a.dtype == b.dtype) and (a.shape == b.shape) and np.array_equal(a, b, equal_nan=True)
                assert same, f"Non-tensor key '{k}' differs"
            non_tensors[k] = v

        new_meta = dict(self.meta_info)
        for k, v in other.meta_info.items():
            if k in new_meta:
                assert new_meta[k] == v, f"Meta key '{k}' differs"
            new_meta[k] = v

        return BatchProto(batch=tensors, non_tensor_batch=non_tensors, meta_info=new_meta)

    @classmethod
    def concat(cls, batches: list["BatchProto"]) -> "BatchProto":
        """Concatenate a list of batches."""
        assert len(batches) > 0, "batches must be non-empty"

        if any(b.batch is not None for b in batches):
            tensors_list = [b.batch for b in batches]
            assert all(t is not None for t in tensors_list), "All batches must have tensor payload when concatenating"
            tensors = torch.cat(tensors_list, dim=0)
        else:
            tensors = None

        # Ensure non_tensor key sets align across batches
        key_sets = [set(b.non_tensor_batch.keys()) for b in batches]
        first_keys = key_sets[0]
        assert all(ks == first_keys for ks in key_sets), "All batches must share identical non_tensor keys for concat"
        keys = first_keys
        non_tensors = {}
        for k in keys:
            # Validate per-batch lengths match len(b)
            parts = []
            for b in batches:
                arr = b.non_tensor_batch[k]
                assert isinstance(arr, np.ndarray), f"Non-tensor '{k}' must be ndarray"
                assert int(arr.shape[0]) == len(b), f"Non-tensor '{k}' dim0={int(arr.shape[0])} != batch len {len(b)}"
                parts.append(arr)
            non_tensors[k] = np.concatenate(parts, axis=0)

        merged_meta = {}
        for b in batches:
            for k, v in b.meta_info.items():
                if k in merged_meta:
                    assert merged_meta[k] == v, f"Conflicting meta_info for key '{k}'"
                else:
                    merged_meta[k] = v

        return cls(batch=tensors, non_tensor_batch=non_tensors, meta_info=merged_meta)

    def repeat(self, repeat_times: int) -> "BatchProto":
        """Repeat the batch by a given number of times."""
        if repeat_times <= 1:
            return BatchProto(batch=self.batch, non_tensor_batch=dict(self.non_tensor_batch), meta_info=self.meta_info)

        # Validate non_tensors align with current batch length before repeating
        for name, arr in self.non_tensor_batch.items():
            assert isinstance(arr, np.ndarray), f"Non-tensor '{name}' must be ndarray"
            assert int(arr.shape[0]) == len(self), f"Non-tensor '{name}' dim0={int(arr.shape[0])} != batch len {len(self)}"

        tensors = None
        if self.batch is not None:
            rep = {k: torch.repeat_interleave(t, repeats=repeat_times, dim=0) for k, t in self.batch.items()}
            tensors = TensorDict(source=rep, batch_size=(len(self) * repeat_times,), device=self.batch.device)

        non_tensors = {k: np.repeat(v, repeat_times, axis=0) for k, v in self.non_tensor_batch.items()}
        return BatchProto(batch=tensors, non_tensor_batch=non_tensors, meta_info=self.meta_info)
    
    def slice(self, start: int | None = None, end: int | None = None, step: int | None = None) -> "BatchProto":
        s = slice(start, end, step)
        tensors = self.batch[s] if self.batch is not None else None
        non_tensors = {k: v[s] for k, v in self.non_tensor_batch.items()}
        return BatchProto(batch=tensors, non_tensor_batch=non_tensors, meta_info=self.meta_info)

    def select(self, idxs: list | np.ndarray | torch.Tensor) -> "BatchProto":
        t = torch.as_tensor(idxs)
        if t.dtype == torch.bool:
            mask = t.view(-1)
            assert int(mask.numel()) == len(self), "Boolean mask length must match batch length"
            idxs_t = torch.nonzero(mask, as_tuple=False).view(-1)
        else:
            idxs_t = t.to(dtype=torch.long).view(-1)

        if self.batch is not None:
            tensors = self.batch[idxs_t]
        else:
            tensors = None
        idxs_np = idxs_t.detach().cpu().numpy()
        non_tensors = {k: v[idxs_np] for k, v in self.non_tensor_batch.items()}
        return BatchProto(batch=tensors, non_tensor_batch=non_tensors, meta_info=self.meta_info)
        