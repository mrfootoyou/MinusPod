"""Tests the split_segments_into_sentences function."""

import math

from utils.text import split_segments_into_sentences, _text_to_sentence_segments

class TestSplitSegmentsIntoSentences:
    def test_does_not_split_one_sentence_segment(self):
        segments = [
            {
                'text': 'Hello world.',
                'start': 3.0,
                'end': 5.0,
                'words': [
                    {'word': ' Hello', 'start': 0.0, 'end': 1.0},
                    {'word': ' world.', 'start': 1.0, 'end': 2.0},
                ]
            }
        ]

        split_segments, split_count = split_segments_into_sentences(segments)
        assert split_count == 0
        assert len(split_segments) == 1
        assert split_segments[0]['text'] == 'Hello world.'

    def test_splits_multi_sentence_segment(self):
        segments = [
            {
                'text': 'Hello world. This. is. a. test.',
                'start': 0.0,
                'end': 5.0,
                'words': [
                    {'word': ' Hello', 'start': 0.0, 'end': 1.0},
                    {'word': ' world.', 'start': 1.0, 'end': 2.0},
                    {'word': ' This.', 'start': 2.0, 'end': 3.0},
                    {'word': ' is.', 'start': 3.0, 'end': 4.0},
                    {'word': ' a.', 'start': 4.0, 'end': 4.5},
                    {'word': ' test.', 'start': 4.5, 'end': 5.0},
                ]
            }
        ]

        split_segments, split_count = split_segments_into_sentences(segments)
        assert split_count == 4
        assert len(split_segments) == 5
        assert split_segments[0]['text'] == 'Hello world.'
        assert split_segments[1]['text'] == 'This.'
        assert split_segments[2]['text'] == 'is.'
        assert split_segments[3]['text'] == 'a.'
        assert split_segments[4]['text'] == 'test.'

    def test_splits_multi_sentence_segment_with_no_end_punctuation(self):
        segments = [
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

        split_segments, split_count = split_segments_into_sentences(segments)
        assert split_count == 1
        assert len(split_segments) == 2
        assert split_segments[0]['text'] == 'Hello world.'
        assert split_segments[1]['text'] == 'This is a test'

    def test_interpolates_when_words_are_missing(self):
        segments = [
            {
                'text': 'One. Two. Three. Four. Five.',
                'start': 0.0,
                'end': 5.0,
            }
        ]

        split_segments, split_count = split_segments_into_sentences(segments, interpolate=True)
        assert split_count == 4
        assert len(split_segments) == 5
        assert split_segments[0]['text'] == 'One.'
        assert split_segments[1]['text'] == 'Two.'
        assert split_segments[2]['text'] == 'Three.'
        assert split_segments[3]['text'] == 'Four.'
        assert split_segments[4]['text'] == 'Five.'

    def test_does_not_split_when_interpolating_single_sentence_segment(self):
        segments = [
            {
                'text': 'Hello world.',
                'start': 3.0,
                'end': 5.0,
            }
        ]

        split_segments, split_count = split_segments_into_sentences(segments)
        assert split_count == 0
        assert len(split_segments) == 1
        assert split_segments[0]['text'] == 'Hello world.'

    def test_does_not_interpolate_when_told_not_to(self):
        segments = [
            {
                'text': 'Hello. world.',
                'start': 3.0,
                'end': 5.0,
            }
        ]

        split_segments, split_count = split_segments_into_sentences(segments, interpolate=False)
        assert split_count == 0
        assert len(split_segments) == 1
        assert split_segments[0]['text'] == 'Hello. world.'

    def test_works_when_only_words_list_is_provided(self):
        segments = [
            {
                # 'text': 'Sentence 1. Sentence 2.',
                # 'start': 0.0,
                # 'end': 2.0,
                'words': [
                    {'word': ' Sentence', 'start': 0.0, 'end': 0.5},
                    {'word': ' 1.', 'start': 0.5, 'end': 1.0},
                    {'word': ' Sentence', 'start': 1.0, 'end': 1.5},
                    {'word': ' 2.', 'start': 1.5, 'end': 2.0},
                ]
            }
        ]

        split_segments, split_count = split_segments_into_sentences(segments)
        assert split_count == 1
        assert len(split_segments) == 2
        assert split_segments[0]['text'] == 'Sentence 1.'
        assert split_segments[0]['start'] == 0.0
        assert split_segments[0]['end'] == 1.0
        assert split_segments[1]['text'] == 'Sentence 2.'
        assert split_segments[1]['start'] == 1.0
        assert split_segments[1]['end'] == 2.0

    def test_supports_multiple_segments(self):
        segments = [
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

        split_segments, split_count = split_segments_into_sentences(segments, interpolate=True)
        assert split_count == 2
        assert len(split_segments) == 4
        assert split_segments[0]['text'] == 'First segment.'
        assert split_segments[1]['text'] == 'Second segment.'
        assert split_segments[2]['text'] == 'Third segment.'
        assert split_segments[3]['text'] == 'Fourth segment.'

    def test_handles_empty_segments_list(self):
        segments = []

        split_segments, split_count = split_segments_into_sentences(segments)
        assert split_count == 0
        assert len(split_segments) == 0

    def test_handles_garbage_input(self):
        segments = [
            {'text': 'foo', 'start': 'bar', 'end': []},
            {'text': 'foo', 'start': 1.0},
            {'words': 123},
            {'foo': 'bar'},
        ]

        for segment in segments:
            split_segments, split_count = split_segments_into_sentences([segment])
            assert split_count == 0
            assert split_segments == [segment]

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

        assert result[0]['text'] == 'Mr. Jones said "Hello Mr."'
        assert result[1]['text'] == 'He is from the U.K. and lives in the U.S.'
        assert result[2]['text'] == 'Do you know why?'
        assert result[3]['text'] == 'He said, "This is a quote and \'this is an inner quote (wow!)\'."'
        assert result[4]['text'] == 'End of the test text'

        assert result[0]['start'] == 1.0
        assert result[0]['end'] > result[0]['start']
        assert math.isclose(result[1]['start'], result[0]['end'] + 0.3, rel_tol=1e-9)
        assert math.isclose(result[2]['start'], result[1]['end'] + 0.3, rel_tol=1e-9)
        assert math.isclose(result[3]['start'], result[2]['end'] + 0.3, rel_tol=1e-9)
        assert math.isclose(result[4]['start'], result[3]['end'] + 0.3, rel_tol=1e-9)
        assert result[4]['end'] == 12.0

        # 'words' array not generated
        assert all(getattr(segment, 'words', None) is None for segment in result)
