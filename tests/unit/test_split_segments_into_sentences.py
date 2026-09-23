"""Tests the split_segments_into_sentences function."""

from utils.text import split_segments_into_sentences

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

        split_segments, split_count = split_segments_into_sentences(segments)
        assert split_count == 4
        assert len(split_segments) == 5
        assert split_segments[0]['text'] == 'One.'
        assert split_segments[0]['start'] == 0.0
        assert split_segments[0]['end'] == 1.0
        assert split_segments[1]['text'] == 'Two.'
        assert split_segments[1]['start'] == 1.0
        assert split_segments[1]['end'] == 2.0
        assert split_segments[2]['text'] == 'Three.'
        assert split_segments[2]['start'] == 2.0
        assert split_segments[2]['end'] == 3.0
        assert split_segments[3]['text'] == 'Four.'
        assert split_segments[3]['start'] == 3.0
        assert split_segments[3]['end'] == 4.0
        assert split_segments[4]['text'] == 'Five.'
        assert split_segments[4]['start'] == 4.0
        assert split_segments[4]['end'] == 5.0

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

        split_segments, split_count = split_segments_into_sentences(segments, do_not_interpolate=True)
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

        split_segments, split_count = split_segments_into_sentences(segments)
        assert split_count == 2
        assert len(split_segments) == 4
        assert split_segments[0]['text'] == 'First segment.'
        assert split_segments[0]['start'] == 0.0
        assert split_segments[0]['end'] == 2.0
        assert split_segments[1]['text'] == 'Second segment.'
        assert split_segments[1]['start'] == 2.0
        assert split_segments[1]['end'] == 4.0
        assert split_segments[2]['text'] == 'Third segment.'
        assert split_segments[2]['start'] == 4.0
        assert split_segments[2]['end'] == 6.0
        assert split_segments[3]['text'] == 'Fourth segment.'
        assert split_segments[3]['start'] == 6.0
        assert split_segments[3]['end'] == 8.0

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
