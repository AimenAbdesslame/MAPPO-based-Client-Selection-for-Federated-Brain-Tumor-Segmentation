
#Block 1: Imports
import torch



#Block 2: Check the inputs
def check_inputs(states, n_patients):
    if not states:
        raise ValueError("states list is empty")
    if len(states) != len(n_patients):
        raise ValueError(f"mismatch: {len(states)} states vs {len(n_patients)} weights")
    if any(n <= 0 for n in n_patients):
        raise ValueError("patient counts must be positive (> 0)")

    ref_keys = states[0].keys()
    for i, s in enumerate(states[1:], 1):
        if s.keys() != ref_keys:
            raise ValueError(f"state {i} keys differ from state 0")
        for k in ref_keys:
            if s[k].shape != states[0][k].shape:
                raise ValueError(f"shape mismatch for key '{k}' between state 0 and {i}")



#Block 3: Compute the average weights
def normalize_weights(n_patients):
    total = sum(n_patients)
    return [n / total for n in n_patients]



#Block 4: Average each tensor
def average_tensors(states, weights):
    avg_state = {}
    ref = states[0]
    for k, ref_t in ref.items():
        if ref_t.is_floating_point():
            acc = torch.zeros(ref_t.shape, dtype=torch.float32)
            for s, w in zip(states, weights):
                acc += w * s[k].to(torch.float32)
            avg_state[k] = acc.to(ref_t.dtype)
        else:
            avg_state[k] = ref_t.clone()
    return avg_state



#Block 5: The main function
def fedavg(states, n_patients):
    check_inputs(states, n_patients)
    weights = normalize_weights(n_patients)
    return average_tensors(states, weights)



#Block 6: Small test
if __name__ == "__main__":
    s1 = {"w": torch.tensor([1.0], dtype=torch.float32), "cnt": torch.tensor([5], dtype=torch.int64)}
    s2 = {"w": torch.tensor([3.0], dtype=torch.float32), "cnt": torch.tensor([10], dtype=torch.int64)}
    s1_orig = s1["w"].clone()

    res = fedavg([s1, s2], [1, 3])

    assert torch.allclose(res["w"], torch.tensor([2.5])), f"Expected 2.5, got {res['w']}"
    assert res["cnt"].item() == 5
    assert torch.equal(s1["w"], s1_orig), "Input state was mutated"
    assert res["w"].dtype == s1["w"].dtype
    assert res["cnt"].dtype == s1["cnt"].dtype

    try:
        fedavg([{"a": torch.tensor([1.0])}, {"b": torch.tensor([1.0])}], [1, 1])
        assert False
    except ValueError:
        pass

    try:
        fedavg([{"a": torch.tensor([1.0])}, {"a": torch.tensor([1.0, 2.0])}], [1, 1])
        assert False
    except ValueError:
        pass

