"""Tests the split_segments_into_sentences function."""

import math

from utils.segments import (
    TextSegment, merge_segments, split_and_merge_segments,
    split_segments_into_sentences, _text_to_sentence_segments,
    )

from .test_segment_normalization import make_text_segment, verify_text_segment


class TestMergeSegments:
    def test_merges_short_segments_with_merge_threshold_1_0(self):
        segments: list[TextSegment] = [
            {'start': 0.0, 'end': 0.5, 'text': 'Short 1.'},
            {'start': 0.5, 'end': 2.0, 'text': 'Long segment.'},
            {'start': 2.0, 'end': 2.3, 'text': 'Short 2.'},
        ]
        merged_segments = merge_segments(segments, maximum_gap=0.01, merge_threshold=1.0, maximum_merge_duration=10.0)
        assert len(merged_segments) == 1
        verify_text_segment(merged_segments[0], 'Short 1. Long segment. Short 2.', 0.0, 2.3)
        assert 'words' not in merged_segments[0]

    def test_merges_short_segments_with_merge_threshold_0_5(self):
        segments: list[TextSegment] = [
            {'start': 0.0, 'end': 0.5, 'text': 'Short 1.'},
            {'start': 0.5, 'end': 2.0, 'text': 'Long segment.'},
            {'start': 2.0, 'end': 2.3, 'text': 'Short 2.'},
        ]
        merged_segments = merge_segments(segments, maximum_gap=0.5, merge_threshold=0.5, maximum_merge_duration=2.0)
        assert len(merged_segments) == 2
        verify_text_segment(merged_segments[0], 'Short 1.', 0.0, 0.5)
        verify_text_segment(merged_segments[1], 'Long segment. Short 2.', 0.5, 2.3)

    def test_merges_short_segments_with_merge_threshold_0_2(self):
        segments: list[TextSegment] = [
            {'start': 0.0, 'end': 0.5, 'text': 'Short 1.'},
            {'start': 0.5, 'end': 2.0, 'text': 'Long segment.'},
            {'start': 2.0, 'end': 2.3, 'text': 'Short 2.'},
        ]
        merged_segments = merge_segments(segments, maximum_gap=0.5, merge_threshold=0.2, maximum_merge_duration=10.0)
        assert len(merged_segments) == 3
        verify_text_segment(merged_segments[0], 'Short 1.', 0.0, 0.5)
        verify_text_segment(merged_segments[1], 'Long segment.', 0.5, 2.0)
        verify_text_segment(merged_segments[2], 'Short 2.', 2.0, 2.3)

    def test_does_not_merge_less_than_two_segments(self):
        segments: list[TextSegment] = [
            {'start': 0.0, 'end': 0.5, 'text': 'Short 1.'},
        ]
        merged_segments = merge_segments(segments, maximum_gap=0.5, merge_threshold=0.01, maximum_merge_duration=10.0)
        assert merged_segments == segments

        merged_segments = merge_segments([], maximum_gap=0.5, merge_threshold=2, maximum_merge_duration=10.0)
        assert merged_segments == []

    def test_merges_short_segments_with_gaps_exceeding_maximum(self):
        segments: list[TextSegment] = [
            {'start': 0.0, 'end': 0.1, 'text': 'Short 1.'},
            # gap of 0.9
            {'start': 1.0, 'end': 2.0, 'text': 'Long segment.'},
            # gap of 0.4
            {'start': 2.4, 'end': 2.5, 'text': 'Short 2.'},
        ]
        merged_segments = merge_segments(segments, maximum_gap=0.5, merge_threshold=2.0, maximum_merge_duration=10.0)
        assert len(merged_segments) == 2
        verify_text_segment(merged_segments[0], 'Short 1.', 0.0, 0.1)
        verify_text_segment(merged_segments[1], 'Long segment. Short 2.', 1.0, 2.5)

    def test_does_not_merge_when_maximum_merge_duration_exceeded(self):
        segments: list[TextSegment] = [
            {'start': 0.0, 'end': 1.0, 'text': 'Short 1.'},
            {'start': 1.0, 'end': 2.0, 'text': 'Short 2.'},
            {'start': 2.0, 'end': 3.0, 'text': 'Short 3.'},
        ]
        merged_segments = merge_segments(segments, maximum_gap=0.5, merge_threshold=10.0, maximum_merge_duration=2.0)
        assert len(merged_segments) == 2
        verify_text_segment(merged_segments[0], 'Short 1. Short 2.', 0.0, 2.0)
        verify_text_segment(merged_segments[1], 'Short 3.', 2.0, 3.0)


class TestSplitSegmentsIntoSentences:
    def test_does_not_split_one_sentence_segment(self):
        segments: list[TextSegment] = [
            {
                'text': '', # not needed when splitting by words
                'start': 0.0,
                'end': 5.0,
                'words': [
                    {'word': ' Hello', 'start': 0.0, 'end': 1.0},
                    {'word': ' world.', 'start': 1.0, 'end': 2.0},
                ]
            }
        ]

        split_segments = split_segments_into_sentences(segments)
        assert len(split_segments) == 1
        verify_text_segment(split_segments[0], 'Hello world.', terminal=True, word_timed_line=True)

    def test_splits_multi_sentence_segment(self):
        segments: list[TextSegment] = [
            {
                'text': '', # not needed when splitting by words
                'start': 0.0,
                'end': 5.0,
                'words': [
                    {'word': ' Hello', 'start': 0.0, 'end': 1.0},
                    {'word': ' world.', 'start': 1.0, 'end': 2.0},
                    {'word': ' This.', 'start': 2.0, 'end': 3.0},
                    {'word': ' Is.', 'start': 3.0, 'end': 4.0},
                    {'word': ' A.', 'start': 4.0, 'end': 4.5},
                    {'word': ' Test.', 'start': 4.5, 'end': 5.0},
                ]
            }
        ]

        split_segments = split_segments_into_sentences(segments)
        assert len(split_segments) == 5
        verify_text_segment(split_segments[0], 'Hello world.', terminal=True)
        verify_text_segment(split_segments[1], 'This.', terminal=True)
        verify_text_segment(split_segments[2], 'Is.', terminal=True)
        verify_text_segment(split_segments[3], 'A.', terminal=True)
        verify_text_segment(split_segments[4], 'Test.', terminal=True)
        assert all(s.get('word_timed_line') == True for s in split_segments)

    def test_does_not_split_honorifics(self):
        segments: list[TextSegment] = [
            {
                'text': '', # not needed when splitting by words
                'start': 0.0,
                'end': 3.0,
                'words': [
                    {'word': ' Hello', 'start': 0.0, 'end': 1.0},
                    {'word': ' Mr.', 'start': 1.0, 'end': 2.0},
                    {'word': ' Turkey.', 'start': 2.0, 'end': 3.0},
                ]
            }
        ]

        split_segments = split_segments_into_sentences(segments)
        assert len(split_segments) == 1
        verify_text_segment(split_segments[0], 'Hello Mr. Turkey.', terminal=True, word_timed_line=True)

    def test_does_not_split_midsentence_abbrev(self):
        segments: list[TextSegment] = [
            {
                'text': '', # not needed when splitting by words
                'start': 0.0,
                'end': 3.0,
                'words': [
                    {'word': ' L.A.', 'start': 0.0, 'end': 1.0},
                    {'word': ' traffic.', 'start': 1.0, 'end': 2.0},
                ]
            }
        ]

        split_segments = split_segments_into_sentences(segments)
        assert len(split_segments) == 1
        verify_text_segment(split_segments[0], 'L.A. traffic.', terminal=True, word_timed_line=True)

    def test_splits_multi_sentence_segment_with_no_end_punctuation(self):
        segments: list[TextSegment] = [
            {
                'text': 'Hello world. This is a test',
                'start': 0.0,
                'end': 5.0,
                'words': [
                    {'word': ' Hello', 'start': 0.0, 'end': 1.0},
                    {'word': ' world.', 'start': 1.0, 'end': 2.0},
                    {'word': ' This', 'start': 2.0, 'end': 3.0},
                    {'word': ' is', 'start': 3.0, 'end': 4.0},
                    {'word': ' a', 'start': 4.0, 'end': 4.5},
                    {'word': ' test', 'start': 4.5, 'end': 5.0},
                ]
            }
        ]

        split_segments = split_segments_into_sentences(segments)
        assert len(split_segments) == 2
        verify_text_segment(split_segments[0], 'Hello world.', terminal=True)
        verify_text_segment(split_segments[1], 'This is a test', terminal=False)
        assert all(s.get('word_timed_line') == True for s in split_segments)
        assert all(s.get('interpolated') is None for s in split_segments)

    def test_interpolates_when_words_are_missing(self):
        segments: list[TextSegment] = [
            {
                'text': 'One. Two. Three. Four. Five.',
                'start': 0.0,
                'end': 5.0,
            }
        ]

        split_segments = split_segments_into_sentences(segments, interpolate=True)
        assert len(split_segments) == 5
        verify_text_segment(split_segments[0], 'One.', terminal=True)
        verify_text_segment(split_segments[1], 'Two.', terminal=True)
        verify_text_segment(split_segments[2], 'Three.', terminal=True)
        verify_text_segment(split_segments[3], 'Four.', terminal=True)
        verify_text_segment(split_segments[4], 'Five.', terminal=True)
        assert all(s.get('interpolated') == True for s in split_segments)

    def test_does_not_split_when_interpolating_single_sentence_segment(self):
        segments: list[TextSegment] = [
            {
                'text': 'Hello world.',
                'start': 3.0,
                'end': 5.0,
            }
        ]

        split_segments = split_segments_into_sentences(segments, interpolate=True)
        assert len(split_segments) == 1
        verify_text_segment(split_segments[0], 'Hello world.', terminal=True, interpolated=None)

    def test_does_not_interpolate_by_default(self):
        segments: list[TextSegment] = [
            {
                'text': 'Hello. world.',
                'start': 3.0,
                'end': 5.0,
            }
        ]

        split_segments = split_segments_into_sentences(segments)
        assert len(split_segments) == 1
        verify_text_segment(split_segments[0], 'Hello. world.', terminal=True, interpolated=None)

    def test_supports_multiple_segments(self):
        segments: list[TextSegment] = [
            {
                'text': 'First segment. Second segment.',
                'start': 0.0,
                'end': 4.0,
            },
            {
                'text': 'Third segment. Fourth segment.',
                'start': 4.0,
                'end': 8.0,
            }
        ]

        split_segments = split_segments_into_sentences(segments, interpolate=True)
        assert len(split_segments) == 4
        verify_text_segment(split_segments[0], 'First segment.', terminal=True)
        verify_text_segment(split_segments[1], 'Second segment.', terminal=True)
        verify_text_segment(split_segments[2], 'Third segment.', terminal=True)
        verify_text_segment(split_segments[3], 'Fourth segment.', terminal=True)
        assert all(s.get('interpolated') == True for s in split_segments)

    def test_handles_empty_segments_list(self):
        split_segments = split_segments_into_sentences([])
        assert len(split_segments) == 0

    def test_text_to_sentence_segments_edge_cases(self):
        text = (
            # Abbreviated titles do not mark the end of a sentence unless it is.
            "Mr. Jones said \"Hello Mr.\" "
            # "U.K." is not the end, "U.S." is.
            "He is from the U.K. and lives in the U.S. "
            # Questions mark the end of a sentence
            "Do you know why? "
            # Quotes within sentences should be handled correctly
            "He said, \"This is a quote and 'this is an inner quote (wow!)'.\" "
            # The final sentence does not require punctuation
            "End of the test text"
        )

        result = _text_to_sentence_segments(text=text, start=1.0, end=12.0, sentence_gap=0.3)
        assert isinstance(result, list)
        assert len(result) == 5
        verify_text_segment(result[0], 'Mr. Jones said "Hello Mr."', terminal=True)
        verify_text_segment(result[1], 'He is from the U.K. and lives in the U.S.', terminal=True)
        verify_text_segment(result[2], 'Do you know why?', terminal=True)
        verify_text_segment(result[3], 'He said, "This is a quote and \'this is an inner quote (wow!)\'."', terminal=True)
        verify_text_segment(result[4], 'End of the test text', terminal=False)
        assert all(s.get('interpolated') == True for s in result)
        assert all(s.get('words') is None for s in result)

        assert result[0]['start'] == 1.0
        assert result[0]['end'] > result[0]['start']
        assert math.isclose(result[1]['start'], result[0]['end'] + 0.3)
        assert math.isclose(result[2]['start'], result[1]['end'] + 0.3)
        assert math.isclose(result[3]['start'], result[2]['end'] + 0.3)
        assert math.isclose(result[4]['start'], result[3]['end'] + 0.3)
        assert result[4]['end'] == 12.0


class TestSplitAndMerge:
    msg: str = ''
    def test_split_and_merge_segments_with_multi_sentence_segments(self):
        segments: list[TextSegment] = [
            make_text_segment('P1 one. P1 two. P1 three.', 0.0, 3.0, make_words=True),
            make_text_segment('P2 one. P2 two. P2 three.', 5.0, 8.0, make_words=True),
        ]
        merged_segments = split_and_merge_segments(segments,
                                                   reporter=lambda m: setattr(self, 'msg', m),
                                                   merge_kwargs={'maximum_gap': 0.3},
                                                   )
        assert len(merged_segments) == 2
        verify_text_segment(merged_segments[0], 'P1 one. P1 two. P1 three.', start=0.0, end=3.0)
        verify_text_segment(merged_segments[1], 'P2 one. P2 two. P2 three.', start=5.0, end=8.0)
        assert self.msg.startswith('Split 2 transcript segments into 6 sentences then merged into 2 segments with maximum gap 0.30s')

    def test_split_and_merge_segments_with_single_sentence_segments(self):
        sentence_gap=0.2
        maximum_gap = 0.3
        segments = _text_to_sentence_segments("One. Two. Three.", 0.0, 3.0, sentence_gap=sentence_gap)
        segments += _text_to_sentence_segments("Four. Five. Six.", 5.0, 8.0, sentence_gap=sentence_gap)
        merged_segments = split_and_merge_segments(
            segments,
            reporter=lambda m: setattr(self, 'msg', m),
            merge_kwargs={'maximum_gap': maximum_gap},
            )
        assert len(merged_segments) == 2
        verify_text_segment(merged_segments[0], 'One. Two. Three.', start=0.0, end=3.0)
        verify_text_segment(merged_segments[1], 'Four. Five. Six.', start=5.0, end=8.0)
        assert self.msg.startswith('Split 6 transcript segments into 6 sentences then merged into 2 segments with maximum gap 0.30s')

    def test_split_and_merge_segments_with_maximum_gap(self):
        maximum_gap = 0.5
        segments: list[TextSegment] = [
            make_text_segment('Short', 0.0, 0.4, make_words=True),
            # 0.1s < maximum_gap (merge)
            make_text_segment('one.', 0.5, 1.0, make_words=True),
            # 0.5s == maximum_gap (merge)
            make_text_segment('Short two.', 1.5, 2.5, make_words=True),
            # 1.0s > maximum_gap (no merge)
            make_text_segment('Short three.', 3.5, 4.0, make_words=True),
        ]
        merged_segments = split_and_merge_segments(segments,
                                                   reporter=lambda m: setattr(self, 'msg', m),
                                                   merge_kwargs={'maximum_gap': maximum_gap},
                                                   )
        assert len(merged_segments) == 2
        verify_text_segment(merged_segments[0], 'Short one. Short two.', start=0.0, end=2.5)
        verify_text_segment(merged_segments[1], 'Short three.', start=3.5, end=4.0)
        assert self.msg.startswith('Split 4 transcript segments into 4 sentences then merged into 2 segments with maximum gap 0.50s')
