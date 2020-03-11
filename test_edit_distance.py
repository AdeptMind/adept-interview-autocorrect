import pytest

from edit_distance import edit_distance


@pytest.mark.parametrize('s1, s2, expected_edits', [
    ('jecket', 'jecket', {}),
    ('jecket', 'jacket', {'substitute_e_a': 1}),
    ('jakcet', 'jacket', {'transpose_c_k': 1}),
    ('jekce', 'jacket', {'substitute_e_a': 1, 'transpose_c_k': 1, 'insert_t': 1}),
    ('aaa', 'bbbb', {'substitute_a_b': 3, 'insert_b': 1}),
    ('bbbb', 'aaa', {'substitute_b_a': 3, 'remove_b': 1}),
])
def test_ed(s1, s2, expected_edits):
    ed, actual_edits = edit_distance(s1, s2, transpositions=True)
    assert actual_edits == expected_edits
    assert ed == sum(expected_edits.values())

