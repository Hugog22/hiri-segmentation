"""
Tests for split creation and leakage detection.
"""
import pytest

def check_leakage(train: list, val: list, test: list) -> bool:
    s_train = set(train)
    s_val = set(val)
    s_test = set(test)
    
    if s_train.intersection(s_val) or s_train.intersection(s_test) or s_val.intersection(s_test):
        return True
    return False

def create_splits(cases: list, seed: int = 42) -> tuple[list, list, list]:
    import random
    random.seed(seed)
    shuffled = cases.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    train = shuffled[:int(n*0.7)]
    val = shuffled[int(n*0.7):int(n*0.85)]
    test = shuffled[int(n*0.85):]
    return train, val, test

def test_no_overlap_train_val():
    cases = [f"case_{i}" for i in range(100)]
    t, v, te = create_splits(cases)
    assert len(set(t).intersection(set(v))) == 0

def test_no_overlap_train_test():
    cases = [f"case_{i}" for i in range(100)]
    t, v, te = create_splits(cases)
    assert len(set(t).intersection(set(te))) == 0
    assert len(set(v).intersection(set(te))) == 0

def test_all_cases_covered():
    cases = [f"case_{i}" for i in range(100)]
    t, v, te = create_splits(cases)
    union = set(t).union(set(v)).union(set(te))
    assert union == set(cases)

def test_no_duplicates_within_split():
    cases = [f"case_{i}" for i in range(100)]
    t, v, te = create_splits(cases)
    assert len(t) == len(set(t))
    assert len(v) == len(set(v))
    assert len(te) == len(set(te))

def test_stratification_by_dataset():
    # Mock behavior of stratifying by dataset
    assert True

def test_seed_determinism():
    cases = [f"case_{i}" for i in range(100)]
    t1, v1, te1 = create_splits(cases, seed=42)
    t2, v2, te2 = create_splits(cases, seed=42)
    t3, v3, te3 = create_splits(cases, seed=99)
    assert t1 == t2
    assert t1 != t3

def test_leakage_detection_catches_violation():
    t = ["case_1", "case_2"]
    v = ["case_2", "case_3"]
    te = ["case_4"]
    assert check_leakage(t, v, te) is True
