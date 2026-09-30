"""Unit tests for the merge_segments function."""

from src.utils.text import merge_segments

class TestMergeSegments:
    def test_merges_short_segments_with_minimum_duration_1_0(self):
        segments = [
            {'start': 0.0, 'end': 0.5, 'text': 'Short 1.', 'words': [' Short', ' 1.']},
            {'start': 0.5, 'end': 2.0, 'text': 'Long segment.', 'words': [' Long', ' segment.']},
            {'start': 2.0, 'end': 2.3, 'text': 'Short 2.', 'words': [' Short', ' 2.']},
        ]
        merged_segments, merge_count = merge_segments(segments, maximum_gap=0.01, minimum_duration=1.0, maximum_duration=10.0)
        assert len(merged_segments) == 1
        assert merge_count == 2
        assert merged_segments[0]['text'] == 'Short 1. Long segment. Short 2.'
        assert merged_segments[0]['start'] == 0.0
        assert merged_segments[0]['end'] == 2.3
        assert merged_segments[0]['words'] == [' Short', ' 1.', ' Long', ' segment.', ' Short', ' 2.']

    def test_merges_short_segments_with_minimum_duration_0_5(self):
        segments = [
            {'start': 0.0, 'end': 0.5, 'text': 'Short 1.'},
            {'start': 0.5, 'end': 2.0, 'text': 'Long segment.'},
            {'start': 2.0, 'end': 2.3, 'text': 'Short 2.'},
        ]
        merged_segments, merge_count = merge_segments(segments, maximum_gap=0.5, minimum_duration=0.5, maximum_duration=10.0)
        assert len(merged_segments) == 2
        assert merge_count == 1
        assert merged_segments[0]['text'] == 'Short 1.'
        assert merged_segments[1]['text'] == 'Long segment. Short 2.'

    def test_merges_short_segments_with_minimum_duration_0_2(self):
        segments = [
            {'start': 0.0, 'end': 0.5, 'text': 'Short 1.'},
            {'start': 0.5, 'end': 2.0, 'text': 'Long segment.'},
            {'start': 2.0, 'end': 2.3, 'text': 'Short 2.'},
        ]
        merged_segments, merge_count = merge_segments(segments, maximum_gap=0.5, minimum_duration=0.2, maximum_duration=10.0)
        assert len(merged_segments) == 3
        assert merge_count == 0
        assert merged_segments[0]['text'] == 'Short 1.'
        assert merged_segments[1]['text'] == 'Long segment.'
        assert merged_segments[2]['text'] == 'Short 2.'

    def test_does_not_merge_less_than_two_segments(self):
        segments = [
            {'start': 0.0, 'end': 0.5, 'text': 'Short 1.'},
        ]
        merged_segments, merge_count = merge_segments(segments, maximum_gap=0.5, minimum_duration=0.01, maximum_duration=10.0)
        assert merged_segments is segments
        assert merge_count == 0

        merged_segments, merge_count = merge_segments([], maximum_gap=0.5, minimum_duration=2, maximum_duration=10.0)
        assert merged_segments == []
        assert merge_count == 0

    def test_merges_short_segments_with_gaps_exceeding_maximum(self):
        segments = [
            {'start': 0.0, 'end': 0.1, 'text': 'Short 1.'},
            # gap of 0.9
            {'start': 1.0, 'end': 2.0, 'text': 'Long segment.'},
            # gap of 0.4
            {'start': 2.4, 'end': 2.5, 'text': 'Short 2.'},
        ]
        merged_segments, merge_count = merge_segments(segments, maximum_gap=0.5, minimum_duration=2.0, maximum_duration=10.0)
        assert len(merged_segments) == 2
        assert merge_count == 1
        assert merged_segments[0]['text'] == 'Short 1.'
        assert merged_segments[1]['text'] == 'Long segment. Short 2.'

    def test_does_not_merge_when_maximum_duration_exceeded(self):
        segments = [
            {'start': 0.0, 'end': 1.0, 'text': 'Short 1.'},
            {'start': 1.0, 'end': 2.0, 'text': 'Short 2.'},
            {'start': 2.0, 'end': 3.0, 'text': 'Short 3.'},
        ]
        merged_segments, merge_count = merge_segments(segments, maximum_gap=0.5, minimum_duration=10.0, maximum_duration=2.0)
        assert len(merged_segments) == 2
        assert merge_count == 1
        assert merged_segments[0]['text'] == 'Short 1. Short 2.'
        assert merged_segments[1]['text'] == 'Short 3.'
