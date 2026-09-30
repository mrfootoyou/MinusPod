import heapq
import math
import re
import string
import numpy as np

from copy import deepcopy
from itertools import count
from typing import Callable, Sequence, TypedDict, NotRequired, TypeGuard, Any, cast


class WordSegment(TypedDict):
    """A typed dictionary representing a word-level segment within a text segment.
    Use `validate_word_segment` to validate its correctness.

    Attributes:
        word (str): The space-prefixed word text. Empty is *not* allowed.
        start (float|int): The start time of the word, in seconds.
        end (float|int): The end time of the word, in seconds. Must be
          greater than or equal to `start`.
    """
    word: str
    start: float|int
    end: float|int

class TextSegment(TypedDict):
    """A typed dictionary representing a segment of text with optional word-level
    timing information. Use `validate_text_segment` to validate its correctness.

    Attributes:
        text (str): The text content of the segment. Empty is allowed.
        start (float|int): The start time of the segment, in seconds.
        end (float|int): The end time of the segment, in seconds. Must be
          greater than or equal to `start`.
        words (NotRequired[list[WordSegment]]): Optional list of word-level
          segments. All words must be ordered chronologically without overlap
          and all must be between the text segment's start and end time (they
          should span the entire duration of the segment).
        terminal (NotRequired[bool]): Optional flag indicating if the segment
          ends with sentence punctuation.
        interpolated (NotRequired[bool]): Optional flag indicating if the segment
          timing is interpolated.
        word_timed_line (NotRequired[bool]): Optional flag indicating if the
          segment is word-timed.
    """
    text: str
    start: float|int
    end: float|int
    words: NotRequired[list[WordSegment]]
    terminal: NotRequired[bool]
    word_timed_line: NotRequired[bool]
    interpolated: NotRequired[bool]


def _segment_duration(seg: WordSegment|TextSegment) -> float:
    return float(seg['end']) - float(seg['start'])

def _segment_gap(left: WordSegment|TextSegment,
                 right: WordSegment|TextSegment) -> float:
    return float(right['start']) - float(left['end'])

def _join_word_segments(word_segments: Sequence[WordSegment]) -> str:
    """Join a list of word segments into a single sentence/utterance.
    The list must contain zero or more normalized space-prefixed word segments;
    space will not be inserted between words.
    """
    return ''.join(w['word'] for w in word_segments).strip()

def _is_non_word(c: str) -> bool:
    return c.isspace() or c in string.punctuation

def _skip_non_word(s: str, i: int) -> int:
    while i < len(s) and _is_non_word(s[i]):
        i += 1
    return i

def _starts_with_end_of(cmp: str, src: str, si: int) -> bool:
    """Check if `cmp` starts with `src[si:]`, ignoring non-word characters."""
    ci = 0
    while si < len(src) and ci < len(cmp):
        if _is_non_word(src[si]) and _is_non_word(cmp[ci]):
            si = _skip_non_word(src, si + 1)
            ci = _skip_non_word(cmp, ci + 1)
            continue
        if src[si] == cmp[ci]:
            si += 1
            ci += 1
            continue
        break # mismatch
    return si == len(src)

def _text_leading_diff(src: str, cmp: str) -> str:
    """
    Subtracts `cmp` from the end of `src`. For example:
    `_text_leading_diff("abc xyz", " xyz...") => "abc"`.
    Returns `src` if there is no overlap, and an empty string if there is no
    difference.
    Ignores case and whitespace and punctuation between words.
    """
    # Remove trailing punctuation from `src` since it often contains
    # trailing punctuation not present in `cmp`
    orig = src
    src = src.lower().rstrip(string.punctuation)
    cmp = cmp.lower()
    n = 0
    while not _starts_with_end_of(cmp, src, n):
        n += 1
    if n >= len(src):
        return orig
    return orig[:n]


def is_text_segment(obj: Any) -> TypeGuard[TextSegment]:
    """Check if an object is compatible with a `TextSegment`.
    Use `validate_text_segment` to validate its correctness.

    Args:
        obj (Any): The object to check.
    Returns:
        bool: True if the object is compatible with a `TextSegment`, False otherwise.
    """
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("text"), str)
        and isinstance(obj.get("start"), (float, int))
        and isinstance(obj.get("end"), (float, int))
        and ((x := obj.get("words")) is None or (isinstance(x, list) and all(is_word_segment(w) for w in x)))
        and ((x := obj.get("terminal")) is None or isinstance(x, bool))
        and ((x := obj.get("interpolated")) is None or isinstance(x, bool))
        and ((x := obj.get("word_timed_line")) is None or isinstance(x, bool))
    )

def validate_text_segment(segment: TextSegment,
                          strict: bool = True) -> None:
    """Validate the semantic correctness of a single text segment.

    If `strict` is True, all word segments are validated in strict mode,
    `start` must be less than or equal to `end`, and all word timings must
    be ordered and exist within [start, end].

    See `validate_word_segments` for details on word segment validation.

    Args:
        segment (TextSegment): The text segment to validate.
        strict (bool, optional): Whether to enforce strict validation rules.
          Defaults to True.
    Raises:
        TypeError: The input is not a compatible `TextSegment`.
        ValueError: The segment has invalid timing or a word segment is invalid
          or out of order.
    """
    if not is_text_segment(segment):
        raise TypeError(f"Invalid text segment structure: {segment}")
    # structure is correct. validate values...
    if not (math.isfinite(segment["start"]) and math.isfinite(segment["end"])):
        raise ValueError(f"Invalid segment timing: {segment}")
    if strict and segment["end"] < segment["start"]:
        raise ValueError(f"Segment end time is before start time: {segment}")
    if (words := segment.get("words")):
        validate_word_segments(words, strict=strict)
        if strict:
            if words[0]["start"] < segment["start"]:
                raise ValueError(f"Word segments starts before the parent segment: {segment}")
            if words[-1]["end"] > segment["end"]:
                raise ValueError(f"Word segments ends after the parent segment: {segment}")
    
def validate_text_segments(segments: Sequence[TextSegment],
                           strict: bool = True) -> None:
    """Validate the semantic correctness of a list of text segments.

    If `strict` is True, all segments are validated in strict mode, and all
    must be sorted by (start, end). Overlap is allowed.

    See `validate_text_segment` for details on individual segment validation.

    Args:
        segments (Sequence[TextSegment]):
          The list of zero or more text segments to validate.
        strict (bool, optional):
          Whether to enforce strict validation rules.
    Raises:
        TypeError: The input is not a compatible list of `TextSegment`.
        ValueError: A segment is invalid or out of order.
    """
    if not isinstance(segments, Sequence):
        raise TypeError(f"Invalid list of text segments: {segments}")
    prev = None
    for segment in segments:
        validate_text_segment(segment, strict=strict)
        if (strict
            and prev is not None
            # overlap is allowed, but the segments must be ordered by (start, end)
            and (segment["start"], segment["end"]) < (prev["start"], prev["end"])):
            raise ValueError(f"Text segments are out of order: {segment}")
        prev = segment


def is_word_segment(obj: Any) -> TypeGuard[WordSegment]:
    """Check if an object is compatible with a `WordSegment`.
    Use `validate_word_segment` to validate its correctness.

    Args:
        obj (Any): The object to check.
    Returns:
        bool: True if the object is compatible with a `WordSegment`, False otherwise.
    """
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("word"), str)
        and isinstance(obj.get("start"), (float, int))
        and isinstance(obj.get("end"), (float, int))
    )

def validate_word_segment(word: WordSegment,
                          strict: bool = True) -> None:
    """Validate the semantic correctness of a single word segment.

    If `strict` is True, then `start` must be less than or equal to `end`,
    and the `word` must be space-prefixed.

    Args:
        word (WordSegment):
          The word segment to validate.
        strict (bool, optional):
          Whether to enforce strict validation rules.
    Raises:
        TypeError: The input is not a compatible `WordSegment`.
        ValueError: The word segment has invalid timing.
    """
    if not is_word_segment(word):
        raise TypeError(f"Invalid word segment structure: {word}")
    # structure is correct. validate values...
    if not (math.isfinite(word["start"]) and math.isfinite(word["end"])):
        raise ValueError(f"Invalid word segment timing: {word}")
    if strict:
        if word["end"] < word["start"]:
            raise ValueError(f"Word segment end time is before start time: {word}")
        if not word["word"]:
            raise ValueError(f"Word segment is empty text: {word}")
        if not word["word"].startswith(" "):
            raise ValueError(f"Word segment is not space-prefixed: {word}")

def validate_word_segments(words: Sequence[WordSegment],
                           strict: bool = True) -> None:
    """Validate a list of word segments.

    If `strict` is True, all word segments are validated in strict mode, and
    all must be ordered by their start times. Overlap is *not* allowed.

    See `validate_word_segment` for details on individual word segment validation.

    Args:
        words (Sequence[WordSegment]):
          The list of zero or more word segments to validate.
        strict (bool, optional):
          Whether to enforce strict validation rules.
    Raises:
        TypeError: The input is not a compatible list of `WordSegment`.
        ValueError: A word segment is invalid or out of order.
    """
    if not isinstance(words, Sequence):
        raise TypeError(f"Invalid list of word segments: {words}")
    prev_end = None
    for word in words:
        validate_word_segment(word, strict=strict)
        if (strict
            and prev_end is not None
            and word["start"] < prev_end):
            raise ValueError(f"Word segments are overlapping or out of order: {words}")
        prev_end = word["end"]


def normalize_segments(segments: Sequence[TextSegment],
                       normalize_from_words: bool = True,
                       deduplicate: bool = True,
                       invalid_segments: list[Any]|None = None,
                       ) -> list[TextSegment]:
    """
    Normalize the given list of segments by:
    1. Ensuring each input is valid using `validate_text_segment(strict=False)`.
    2. Normalizing each individual segment (see `normalize_segment`).
    3. Sort all segments by their `start` and `end` time.

    Note: The segments are modified in-place.

    Args:
      segments (Sequence[TextSegment]):
        The input list of segments to normalize.
      normalize_from_words (bool):
        Whether to reconstruct the segment text and timings from its word
        segments if present.
      deduplicate (bool):
        Whether to remove duplicate segments after normalization.
      invalid_segments (list[Any]|None):
        Optional list to collect invalid segments. If provided, invalid
        segments will be appended to this list instead of raising an exception.
    Returns:
      list[TextSegment]: A new list of the normalized text segments.
    Exceptions:
      TypeError: The input is not a compatible list of `TextSegment`.
      ValueError: The input contains an invalid value.
    """
    # Validate the input before normalization.
    if not isinstance(segments, Sequence):
        raise TypeError(f"Invalid sequence of text segments: {segments}")
    valid_segments: list[TextSegment] = []
    for s in segments:
        try:
            validate_text_segment(s, strict=False)
            valid_segments.append(s)
        except (TypeError, ValueError):
            if invalid_segments is not None:
                invalid_segments.append(s)
            else:
                raise
    return _normalize_segments(valid_segments,
                               normalize_from_words=normalize_from_words,
                               deduplicate=deduplicate,
                               bad_segments=invalid_segments)

def _normalize_segments(segments: list[TextSegment],
                        normalize_from_words: bool = True,
                        deduplicate: bool = True,
                        bad_segments: list[Any]|None = None,
                        ) -> list[TextSegment]:
    # normalize all segments
    segments = [
        _normalize_segment(s, 
                           normalize_from_words=normalize_from_words,
                           deduplicate=deduplicate)
        for s in segments
    ]

    # sort by start and end times
    segments.sort(key=lambda s: (s['start'], s['end']))

    if deduplicate and len(segments) >= 2:
        # we're sorted by start and end time.
        # Look for these scenarios:
        # #1: s0 contained in s1. Skip s0.
        #   s0----->....|
        #   s1--------->|
        # #2: s1 contained in s0. Skip s1
        #   s0--------->|
        #     s1--->....|
        # #3: Partial overlap. Take s0-s1 and s1
        #   s0------>|
        #       s1----->|
        # #4: No overlap, take s0 and s1
        #   s0---->|
        #           s1---->|
        unique_segments = []
        i0, i1 = 0, 1
        while i1 < len(segments):
            s0 = segments[i0]
            s1 = segments[i1]
            s0_start = s0['start']
            s0_end = s0['end']
            s1_start = s1['start']
            s1_end = s1['end']
            # fact: s0_start <= s1_start
            if s0_start == s1_start:
                # fact: s0_end <= s1_end
                # scenario #1: full overlap. skip s0, advance to s1
                i0 = i1
            # fact: s0_start < s1_start
            elif s1_end <= s0_end:
                # scenario #2: full overlap. skip s1, stay on s0
                pass
            # fact: s1_end < s0_end
            elif s1_start < s0_end:
                # scenario #3: partial overlap. take {s0-s1}, advance to s1
                diff: TextSegment = {
                    **s0, # inherit all attr
                    'end': s1_start,
                }
                if (words := s0.get('words')) is not None:
                    words = [w for w in words if w['start'] < s1_start]
                    diff['words'] = words
                    if words:
                        diff['end'] = words[-1]['end']
                        if normalize_from_words:
                            diff['text'] = _join_word_segments(words)
                if not words or not normalize_from_words:
                    diff['text'] = _text_leading_diff(s0['text'], s1['text']).strip()
                if diff['text']:
                    diff['terminal'] = _has_eos(diff['text'])
                    unique_segments.append(diff)
                i0 = i1
            else:
                # scenario #4: no overlap. take s0, advance to s1
                unique_segments.append(s0)
                i0 = i1
            i1 += 1
        unique_segments.append(segments[i0])
        segments = unique_segments

    return segments


def normalize_segment(segment: TextSegment,
                      normalize_from_words: bool = True,
                      deduplicate: bool = True,
                      ) -> TextSegment:
    """
    Normalize the given segment by:
    1. Ensuring input is valid using `validate_text_segment(strict=False)`.
    2. Trimming whitespace from `text`.
    3. Converting `start` and `end` to floats.
    4. Clip `end` to be greater than or equal to `start`.
    5. Normalizing word segments if present (see `normalize_word_segments`).
    6. Reconstructing the segment text and timings from its word segments
       if `normalize_from_words` is True and it has word segments.

    Note: The segment is modified in-place.

    Args:
      segment (TextSegment):
        The text segment to normalize.
      normalize_from_words (bool):
        Whether to reconstruct the segment text and timings from its word
        segments if present.
      deduplicate (bool):
        Whether to remove duplicate word segments.
    Returns:
      TextSegment: The normalized text segment.
    Exceptions:
      TypeError: The input is not a compatible `TextSegment`.
      ValueError: The input contains an invalid value.
    """
    # Validate the input before normalization.
    validate_text_segment(segment, strict=False)
    return _normalize_segment(segment,
                              normalize_from_words=normalize_from_words,
                              deduplicate=deduplicate)

def _normalize_segment(segment: TextSegment,
                       normalize_from_words: bool = True,
                       deduplicate: bool = True,
                       ) -> TextSegment:
    # trim text
    segment['text'] = segment['text'].strip()
    # ensure start and end are floats
    start = float(segment['start'])
    end = max(start, float(segment['end']))
    segment['start'] = start
    segment['end'] = end
    # normalize word segments if present
    if (words := segment.get('words')):
        words = _normalize_word_segments(words, deduplicate=deduplicate)
        segment['words'] = words
        if normalize_from_words:
            segment['text'] = _join_word_segments(words)
            segment['start'] = words[0]['start']
            segment['end'] = words[-1]['end']
    return segment


def normalize_word_segments(segments: Sequence[WordSegment],
                            deduplicate: bool = True,
                            invalid_segments: list[Any]|None = None,
                            ) -> list[WordSegment]:
    """
    Normalize the given segments by:
    1. Ensuring each input is valid using `validate_word_segment(strict=False)`.
    2. Normalizing each individual word segment (see `normalize_word_segment`).
    3. Sort all word segments by their `start` time.
    4. Merging any split words, which is common in Whisper transcripts.
    5. Ensure all words have a single leading space, including the first word.

    Note: The word segments are modified in-place.

    Args:
      input (Sequence[WordSegment]): The input list of word segments.
      deduplicate (bool): Whether to remove duplicate word segments.
      invalid_segments (list[Any]|None):
        Optional list to collect invalid segments. If provided, invalid
        segments will be appended to this list instead of raising an exception.
    Returns:
      list[WordSegment]: A new list of normalized word segments.
    Exceptions:
      TypeError: The input is not a compatible list of `WordSegment`.
      ValueError: The input contains an invalid value.
    """
    # Validate the input before normalization.
    if not isinstance(segments, Sequence):
        raise TypeError(f"Invalid sequence of word segments: {segments}")
    valid_segments: list[WordSegment] = []
    for s in segments:
        try:
            validate_word_segment(s, strict=False)
            valid_segments.append(s)
        except (TypeError, ValueError):
            if invalid_segments is not None:
                invalid_segments.append(s)
            else:
                raise
    return _normalize_word_segments(valid_segments,
                                    deduplicate=deduplicate)

def _normalize_word_segments(input: list[WordSegment],
                             deduplicate: bool = True,
                             ) -> list[WordSegment]: 
    if not input:
        return []
    # normalize each word segment individually
    words = [_normalize_word_segment(word,
                                     ensure_space_prefix=False, # we'll do this below
                                     )
             for word in input]

    # sort by start time. this is a stable sort and will not reorder words
    # with the same start time.
    words.sort(key=lambda w: w['start'])

    # merge all split words if it appears to be space-prefixed
    any_space_prefixed = any(word['word'].startswith(' ') for word in words)
    if any_space_prefixed:
        # Words without a leading space are continuation of the previous word.
        # Here are some examples seen in real Whisper transcripts
        # ('|' indicates the split location):
        #  " U|.S."
        #  " 4|.6"
        #  " example|.com"
        #  " 100|,000"
        #  " y|'all"
        #  " A|$AP"
        #  " 90|%"
        #  " AT|&T"
        #  " A|+"
        #  " built|-in"
        last_idx, idx = 0, 1
        last = words[0]
        while idx < len(words):
            word = words[idx]
            idx += 1
            text = word['word']
            if not text.startswith(' '): # also matches empty strings
                # merge with the previous word
                last['word'] += text
                last['end'] = max(last['end'], word['end'])
                continue
            last_idx += 1
            words[last_idx] = last = word
        while last_idx + 1 < len(words): words.pop() # remove leftovers

    # ensure all words have a leading space, including the first word
    for word in words:
        word['word'] = " " + word['word'].strip()

    if deduplicate and len(words) >= 2:
        # remove consecutive duplicates
        last_idx, idx = 0, 1
        last = words[0]
        while idx < len(words):
            word = words[idx]
            idx += 1
            if word['start'] == last['start'] and word['word'] == last['word']:
                # duplicate word. skip it
                continue
            # adjust last end time (if needed) to ensure no overlap.
            last['end'] = min(last['end'], word['start'])
            last_idx += 1
            words[last_idx] = last = word
        while last_idx + 1 < len(words): words.pop() # remove leftovers

    return words


def normalize_word_segment(word: WordSegment,
                           ensure_space_prefix: bool = True,
                           ) -> WordSegment:
    """
    Normalize the given segment by:
    1. Ensuring the input is valid using `validate_word_segment(strict=False)`.
    2. Converting `start` and `end` to floats.
    3. Ensuring `end` is greater than or equal to `start`.
    4. Ensuring that word has a leading space when `ensure_space_prefix` is True.

    Note: The word segment is modified in-place.

    Args:
        word (WordSegment):
          The word segment to normalize.
        ensure_space_prefix (bool):
          Whether to ensure the word has a leading space.
    Returns:
        WordSegment: The normalized word segment. Same as the input `word`.
    Exceptions:
        TypeError: The input is not a compatible `WordSegment`.
        ValueError: The input contains an invalid value.
    """
    # Validate the input before normalization.
    validate_word_segment(word, strict=False)
    return _normalize_word_segment(word, # type: ignore
                                   ensure_space_prefix=ensure_space_prefix)

def _normalize_word_segment(word: WordSegment,
                            ensure_space_prefix: bool = True,
                            ) -> WordSegment:
    # ensure the word has a leading space
    if ensure_space_prefix:
        word['word'] = " " + word['word'].strip()
    # ensure start and end are floats
    start = float(word['start'])
    end = max(start, float(word['end']))
    word['start'] = start
    word['end'] = end
    return word


# End-of-sentence punctuation and quote characters for sentence segmentation.
# U+ff1f: Fullwidth Question Mark
# U+ff01: Fullwidth Exclamation Mark
# U+2018: Left Single Quotation Mark
# U+2019: Right Single Quotation Mark
_EOS_PUNCTUATION = r'[.?!¿¡\uff1f\uff01]'
_END_QUOTES = r'["”»“\'\u2018\u2019]'
# Pattern to match straightforward end-of-sentence punctuation.
# Note: By itself, this will match abbreviations and honorifics.
_EOS_PATTERN = rf'{_EOS_PUNCTUATION}{_END_QUOTES}*$'
# Pattern to match words which DO NOT represent end-of-sentence when followed
# by a period+space. These are usually honorifics, like "Dr" in "Dr. Smith".
# Note: A period followed by a non-space ("i.e.,") or space+lowercase letter
# ("U.S. beef") is never EOS.
_NOT_EOS_WORDS = '|'.join([
    'Mr', 'Mrs', 'Ms',
    'Dr', 'Prof', 'Rev', 'St', 'Hon',
    'Pres', 'Sen', 'Rep', 'Gov', 'Amb', 'Sec', 'Treas',
    'Lt', 'Col', 'Gen', 'Maj', 'Capt', 'Sgt', 'Cpl', 'Pvt',
])
# End-of-sentence pattern for splitting text into sentences.
# Only 'eos' labeled matches are considered sentence boundaries.
# Note: Initialed names like "J.P. Morgan" will be split at the space
# since the initials look like an eos abbreviation ("...in L.A. They...").
_EOS_BOUNDARY_PATTERN = (
    # - Ignore period followed by space+lowercase letter
    rf'(?:[.](?=[ ]+[a-z]))|'
    # - Ignore non-EOS words, like honorifics
    rf'(?:\b(?:{_NOT_EOS_WORDS})[.](?=[ ]))|'
    # - Match all other sentence-ending punctuation followed by optional
    #   quotes, mandatory space, and a non-lowercase letter.
    rf'(?P<eos>{_EOS_PUNCTUATION}{_END_QUOTES}*(?=\s+[^a-z]))'
)

def _has_eos(text: str) -> bool:
    """Check if the given text ends with end-of-sentence punctuation."""
    return re.search(_EOS_PATTERN, text) is not None

def _split_sentences(text: str) -> list[str]:
    """Split the given text into one or more sentences.
    Returns:
      list[str]: a list of trimmed sentences extracted from the input text.
        Always returns at least one sentence.
    """
    text = text.strip()
    if not text:
        return ['']
    sentences = []
    start_idx = 0
    for m in re.finditer(_EOS_BOUNDARY_PATTERN, text, flags=re.DOTALL):
        if (eos := m.end('eos')) >= 0:
            if (sentence := text[start_idx:eos].strip()):
                sentences.append(sentence)
            start_idx = eos
    if start_idx < len(text):
        if (sentence := text[start_idx:].strip()):
            sentences.append(sentence)
    return sentences

def _is_eos_word(word: str, next_word: str|None = None) -> bool:
    """Check if the given space-prefixed word is an end-of-sentence,
    considering the next word to avoid false positives.
    """
    if word.endswith('.'):
        # words ending with a period may generate false positives when
        # they are part of an abbreviation or honorific. In these cases we need
        # to consider the next word as well.
        # Note: these words are expected to be space-prefixed (when appropriate)
        # thus we do not add extra spaces ourselves.
        if next_word:
            # we could use `_split_sentences(word + next_word)[0] == word.strip()`
            # but that is overkill for our purposes.
            m = re.search(_EOS_BOUNDARY_PATTERN, word + next_word, flags=re.DOTALL)
            return m is not None and m.end('eos') == len(word)
        return True
    return _has_eos(word)

def _text_to_sentence_segments(text: str, start: float, end: float,
                               sentence_gap: float,
                               ) -> list[TextSegment]:
    """Convert `text` into sentence-level segments with interpolated
    `start` and `end` times.
    """
    sentences = [
        (sentence, len(sentence.split()))
        for sentence in _split_sentences(text)
    ]

    sentence_segments: list[TextSegment] = []
    if (len(sentences) == 1):
        sentence, _ = sentences[0]
        sentence_segments.append({
            'text': sentence,
            'start': start,
            'end': end,
            'terminal': _has_eos(sentence),
        })
    else:
        # interpolate sentence start and end times based on word counts
        # and the specified sentence gap between sentences
        sentence_start = start
        word_total = sum(word_count for _, word_count in sentences)
        word_delta = (end - start - sentence_gap * (len(sentences) - 1)) / word_total
        if word_delta <= 0.010: # probably an unrealistic unit test
            word_delta = (end - start) / word_total
        for sentence, word_count in sentences:
            sentence_end = sentence_start + word_count * word_delta
            sentence_segments.append({
                'text': sentence,
                'start': round(sentence_start, 3),
                'end': round(sentence_end, 3),
                'interpolated': True,
                'terminal': _has_eos(sentence),
            })
            sentence_start = sentence_end + sentence_gap

    if sentence_segments: # preserve exact start/end
        sentence_segments[0]['start'] = start
        sentence_segments[-1]['end'] = end

    validate_text_segments(sentence_segments)
    return sentence_segments

def _split_segment_into_sentences(seg: TextSegment,
                                  interpolate: bool = False,
                                  sentence_gap: float = 0.2,
                                  max_word_gap: float = 0.5,
                                  ) -> list[TextSegment]:
    """See `split_segments_into_sentences`.
    The segment must be valid and normalized.
    """
    words = seg.get('words')
    if not words:
        if interpolate:
            return _text_to_sentence_segments(seg['text'], seg['start'], seg['end'], sentence_gap)

        sentence_segment: TextSegment = {
                'text': seg['text'],
                'start': seg['start'],
                'end': seg['end'],
                'terminal': _has_eos(seg['text']),
        }
        validate_text_segment(sentence_segment)
        return [sentence_segment]

    # split the segment into sentences based on word boundaries and gaps
    sentences: list[TextSegment] = []
    first_word, next_word = 0, 0
    while next_word < len(words) - 1: # we handle the last word/sentence below
        word_seg = words[next_word]
        word: str = word_seg.get('word', '')

        next_seg = words[next_word+1]
        eos = _is_eos_word(word, next_seg['word'])
        word_gap = _segment_gap(word_seg, next_seg)

        if eos or word_gap > max_word_gap:
            # found end of sentence
            sentence = words[first_word:next_word+1]
            sentences.append({
                'text': _join_word_segments(sentence),
                'start': sentence[0].get('start'),
                'end': sentence[-1].get('end'),
                'words': sentence,
                'word_timed_line': True,
                'terminal': eos,
            })
            first_word = next_word + 1
        next_word += 1

    # handle the last word/sentence
    sentence = words[first_word:]
    sentences.append({
        'text': _join_word_segments(sentence),
        'start': sentence[0].get('start'),
        'end': sentence[-1].get('end'),
        'words': sentence,
        'word_timed_line': True,
        'terminal': _is_eos_word(sentence[-1]['word']),
    })

    # preserve the original start and end times of the segment
    sentences[0]['start'] = seg['start']
    sentences[-1]['end'] = seg['end']

    validate_text_segments(sentences)
    return sentences

def split_segments_into_sentences(segments: Sequence[TextSegment],
                                  interpolate: bool = False,
                                  sentence_gap: float = 0.2,
                                  max_word_gap: float = 0.5,
                                  ) -> list[TextSegment]:
    """
    Splits the given segments into individual sentences based on clear
    punctuation. Each input segment is assumed to contain at least one
    complete sentence, thus segments will only be split and never merged.
    Args:
        segments (Sequence[TextSegment]):
          Zero or more normalized text segments, preferably containing
          word-level timing information.
        interpolate (bool):
          If True and the `words` key is not present, the function will
          interpolate sentence timing from the segment's `text`, `start`, and
          `end` keys.
        sentence_gap (float):
          The gap duration, in seconds, to insert between interpolated
          sentences. Used only when interpolating.
        max_word_gap (float):
          The maximum gap duration, in seconds, between words within the same
          sentence. A gap larger than this indicates a sentence break,
          regardless of punctuation. Only used when `words` key is present in
          the segment.
    Returns:
        list[TextSegment]: A list of segments split into individual sentences.
    Exceptions:
        ex: Input segments are not normalized.
    """
    return [
        sentence
        for seg in segments
        for sentence in _split_segment_into_sentences(
            seg,
            interpolate=interpolate,
            sentence_gap=sentence_gap,
            max_word_gap=max_word_gap,
        )
    ]

def merge_segments(segments: Sequence[TextSegment],
                   maximum_gap: float,
                   merge_threshold: float = 30.0,
                   maximum_merge_duration: float = 30.0,
                   ) -> list[TextSegment]:
    """
    Merge all segments with a duration less than `merge_threshold`
    with its shortest near neighbor. Segments will only be merged if
    the gap between them is less than `maximum_gap` and the merged
    segment does not exceed `maximum_merge_duration`.
    Args:
      segments (list[TextSegment]):
        Zero or more normalized text segments.
      maximum_gap (float):
        The maximum allowed gap (in seconds) between segments to consider
        them for merging. Segments separated by a gap larger than this will
        not be merged. Smaller values result in fewer merges while larger
        values may merge segments that are less closely related.
      merge_threshold (float):
        Segments shorter than this duration will be merged with their
        shortest nearest neighbor.
      maximum_merge_duration (float):
        The maximum length of merged segments (in seconds). If a merged
        segment would exceed this duration, it will not be merged.
    Returns:
      list[TextSegment]:
        A cloned list of segments with short segments merged according to the
        specified criteria.
    Exceptions:
      ex: Input `segments` are not normalized.
    """
    # make a copy of all segments
    segments = deepcopy([*segments])

    original_len = len(segments)
    if len(segments) <= 1 or merge_threshold <= 0.0:
        return segments

    # Algorithm:
    # 1. Find the shortest segment less than `merge_threshold`.
    # 2. Find its nearest neighbor less than `maximum_gap` distance away
    #    AND whose merged duration is less than `maximum_merge_duration`.
    # 3. Merge it with the chosen neighbor if available.

    # use a priority queue (heap) to keep track of the shortest segments
    short_segments: list[tuple[float, int, TextSegment]] = []
    tie_breaker = count()
    def queue_short_segment(seg: TextSegment):
        duration = _segment_duration(seg)
        if duration <= merge_threshold:
            # tie-breaker ensures that heapq never compares the segments
            # when there are duplicate durations
            heapq.heappush(short_segments, (duration, next(tie_breaker), seg))

    def next_shortest_segment() -> tuple[TextSegment, float] | None:
        if not short_segments:
            return None
        duration, _, shortest = heapq.heappop(short_segments)
        return (shortest, duration)

    def remove_from_queue(seg: TextSegment):
        # not the most efficient method, but heapq does not support direct
        # removal of arbitrary elements
        for i, (_, _, s) in enumerate(short_segments):
            if s is seg:
                short_segments.pop(i)
                heapq.heapify(short_segments)
                return

    # add all [short] segments to the priority queue
    for seg in segments:
        queue_short_segment(seg)

    # 1. get next shortest segment
    while (next_short := next_shortest_segment()):
        shortest, duration = next_short

        # 2. find neighboring merge candidates...
        i = segments.index(shortest)
        left = segments[i - 1] if i > 0 else None
        left_gap = _segment_gap(left, shortest) if left else math.inf
        if left and (left_gap > maximum_gap or _segment_duration(left) > maximum_merge_duration - duration):
            # cannot merge with left
            left, left_gap = None, math.inf

        right = segments[i + 1] if (i + 1) < len(segments) else None
        right_gap = _segment_gap(shortest, right) if right else math.inf
        if right and (right_gap > maximum_gap or _segment_duration(right) > maximum_merge_duration - duration):
            # cannot merge with right
            right, right_gap = None, math.inf

        # 3. merge with the nearer neighbor...
        merged = None
        if left and left_gap <= right_gap:
            # append to left
            left['text'] = left['text'] + ' ' + shortest['text']
            if shortest['end'] > left['end']:
                left['end'] = shortest['end']
            if left.get('words') or shortest.get('words'):
                left['words'] = left.get('words', []) + shortest.get('words', [])
            left.pop('terminal', None)
            if shortest.get('terminal') is not None:
                left['terminal'] = shortest['terminal'] # type: ignore
            merged = left
        elif right:
            # prepend to right
            right['text'] = shortest['text'] + ' ' + right['text']
            if shortest['start'] < right['start']:
                right['start'] = shortest['start']
            if right.get('words') or shortest.get('words'):
                right['words'] = shortest.get('words', []) + right.get('words', [])
            merged = right
        if merged:
            segments.remove(shortest)
            remove_from_queue(merged)
            queue_short_segment(merged)

    return segments

def get_optimal_maximum_gap(sentences: Sequence[TextSegment], 
                            target_percentile: float = 50.0,
                            max_gap_threshold: float = 3.0,
                            ) -> float:
    """
    Calculates the optimal maximum gap between consecutive sentences based on
    the specified percentile.
    Args:
        sentences (Sequence[TextSegment]):
          List of normalized text segments.
        target_percentile (float, optional):
          The percentile to use for determining the maximum gap.
        max_gap_threshold (float, optional):
          Maximum gap duration to consider.
    Returns:
        float: The gap at the target percentile.
    """
    if not sentences or len(sentences) == 1:
        return 0.5
    starts = np.array([s['start'] for s in sentences])
    ends = np.array([s['end'] for s in sentences])
    gaps = starts[1:] - ends[:-1]
    gaps = gaps[(gaps >= 0.0) & (gaps <= max_gap_threshold)]
    return float(np.percentile(gaps, target_percentile))

def split_and_merge_segments(segments: Sequence[TextSegment],
                             reporter: Callable[[str], None]|None = None,
                             split_kwargs: dict = {},
                             merge_kwargs: dict = {},
                             ) -> list[TextSegment]:
    """
    Splits the given text segments into sentences and then merges them into
    paragraphs using default settings.
    Args:
        segments (Sequence[TextSegment]):
          List of text segments to process.
        maximum_gap (float, optional):
          The maximum gap to use when merging segments. If None, it will be
          calculated using `get_optimal_maximum_gap`.
        reporter (Callable[[str], None]|None, optional):
          Callable instance for reporting summary.
        split_kwargs (dict, optional):
          Additional arguments to pass to `split_segments_into_sentences`.
        merge_kwargs (dict, optional):
          Additional arguments to pass to `merge_segments`.
    Returns:
        list[TextSegment]: List of split and merged text segments.
    """
    split_kwargs = {} if split_kwargs is None else split_kwargs
    merge_kwargs = {} if merge_kwargs is None else merge_kwargs

    sentence_segments = split_segments_into_sentences(segments, **split_kwargs)
    if merge_kwargs.get('maximum_gap') is None:
        merge_kwargs['maximum_gap'] = get_optimal_maximum_gap(sentence_segments)
    merged_segments = merge_segments(sentence_segments, **merge_kwargs)
    if reporter:
        reporter(f"Split {len(segments)} transcript segments into {len(sentence_segments)} sentences then merged into {len(merged_segments)} segments with maximum gap {merge_kwargs['maximum_gap']:.2f}s")
    return merged_segments
