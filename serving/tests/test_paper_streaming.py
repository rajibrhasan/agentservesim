from copy import deepcopy

import pytest

from experiments.paper.streaming import USER_HEADER, convert


def source():
    first = [10, 11]
    second = first + [20, 21] + [128009] + USER_HEADER + [30]
    return [{'program_id': 'p', 'turns': [
        dict(turn_idx=0, input_tok_ids=first, input_toks=len(first),
             output_toks=2, tool_ms=100),
        dict(turn_idx=1, input_tok_ids=second, input_toks=len(second),
             output_toks=0, collection_output_toks=3, tool_ms=0),
    ]}]


def test_streaming_and_materialized_traces_retain_identical_context_lengths():
    real, sim = convert(source(), {'p': 123})
    r, s = real[0]['sub_requests'], sim[0]['sub_requests']
    assert real[0]['arrival_time_ns'] == sim[0]['arrival_time_ns'] == 123
    context = []
    for delta, full in zip(r, s):
        context += delta['input_tok_ids']
        assert context == full['input_tok_ids']
        assert full['input_toks'] == len(context)
        assert delta['output_toks'] == len(full['output_tok_ids'])
        assert delta['tool_duration_ns'] == full['tool_duration_ns']
        context += full['output_tok_ids']
    assert r[1]['output_toks'] == 3
    assert r[1]['input_tok_ids'] == [128009] + USER_HEADER + [30]
    assert all(token >= 0 for turn in r for token in turn['input_tok_ids'])


@pytest.mark.parametrize('offset', [0, 4])
def test_rejects_unverified_source_boundaries(offset):
    data = deepcopy(source())
    data[0]['turns'][1]['input_tok_ids'][offset] = 999
    with pytest.raises(ValueError):
        convert(data, {'p': 123})


def test_replay_caps_change_generation_without_changing_source_tool_boundary():
    selected = {'p': [dict(output_toks=1, tool_duration_ns=321),
                      dict(output_toks=2, tool_duration_ns=0)]}
    real, sim = convert(source(), {'p': 123}, selected)
    r, s = real[0]['sub_requests'], sim[0]['sub_requests']
    assert [t['output_toks'] for t in r] == [1, 2]
    assert r[0]['tool_duration_ns'] == 321
    assert r[1]['input_tok_ids'] == [128009] + USER_HEADER + [30]
    assert s[1]['input_toks'] == 2 + 1 + len(r[1]['input_tok_ids'])
