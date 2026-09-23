"""Text utility functions.

Provides shared transcript text extraction functions.
"""

import heapq
import math
import re

from itertools import count
from utils.time import parse_timestamp

# Edge-proximity tolerance for cut/trim boundaries. Used by the
# pattern-rewrite anchor gate (a large trimmed boundary must land within
# this distance of a transcript segment edge to be trusted) and as the base
# for the ad reviewer's prose-mismatch warning margin. Boundaries are never
# moved to segment edges; this only measures proximity.
BOUNDARY_SNAP_TOLERANCE_S = 3.0


def word_boundary_re(terms) -> re.Pattern | None:
    """One case-insensitive alternation matching any of `terms` as whole words,
    or None when nothing is left to match. Longest first so an alternative that
    prefixes another cannot win the match.

    Lookarounds, not \\b: a brand whose first or last character is not a word
    character ("Liquid I.V.", "Yahoo!") has no word boundary there, and \\b
    would never match it.
    """
    escaped = sorted({t.strip() for t in terms if t and t.strip()},
                     key=len, reverse=True)
    if not escaped:
        return None
    return re.compile(
        r'(?<!\w)(?:' + '|'.join(re.escape(t) for t in escaped) + r')(?!\w)',
        re.IGNORECASE)


def pattern_offsets(text: str, patterns: dict) -> dict[str, list[int]]:
    """Match offsets of each named pattern in `text`, by name. Names with no
    match are left out, so len() counts the names the text carries."""
    found: dict[str, list[int]] = {}
    for name, pattern in patterns.items():
        offsets = [match.start() for match in pattern.finditer(text)]
        if offsets:
            found[name] = offsets
    return found


def truncate(text: str, limit: int) -> str:
    """Cut text to limit characters, ellipsis included in the count."""
    if not text or len(text) <= limit:
        return text
    # No room for the ellipsis: text[:limit - 3] would slice from the end.
    if limit <= 3:
        return text[:max(limit, 0)]
    return text[:limit - 3].rstrip() + '...'


def parse_transcript_segments(transcript_text: str) -> list[dict]:
    """Parse VTT-formatted transcript text into segment dicts.

    Parses lines in the format:
    [HH:MM:SS.mmm --> HH:MM:SS.mmm] Text content here

    Args:
        transcript_text: Raw transcript string with timestamped lines

    Returns:
        List of dicts with 'start', 'end', 'text' keys
    """
    segments: list[dict] = []
    for line in transcript_text.split('\n'):
        if line.strip() and line.startswith('['):
            try:
                time_part, text_part = line.split('] ', 1)
                time_range = time_part.strip('[')
                start_str, end_str = time_range.split(' --> ')
                segments.append({
                    'start': parse_timestamp(start_str),
                    'end': parse_timestamp(end_str),
                    'text': text_part,
                })
            except (ValueError, TypeError):
                continue
    return segments


def get_transcript_text_for_range(
    segments: list[dict],
    start_time: float,
    end_time: float,
) -> str:
    """Get concatenated transcript text for a time range.

    Args:
        segments: List of transcript segment dicts with 'start', 'end', 'text'
        start_time: Start of range in seconds
        end_time: End of range in seconds

    Returns:
        Concatenated text from all overlapping segments
    """
    texts = []
    for seg in segments:
        if seg['end'] >= start_time and seg['start'] <= end_time:
            texts.append(seg.get('text', ''))
    return ' '.join(texts)


def get_timestamped_transcript_for_range(
    segments: list[dict],
    start_time: float,
    end_time: float,
) -> str:
    """Get per-segment timestamped transcript lines for a time range.

    Unlike get_transcript_text_for_range, which strips intra-span timestamps,
    this keeps each overlapping segment on its own line with its start/end in
    seconds so every sentence carries its boundary (reviewer prompts need
    these anchors to emit exact trim timestamps).

    Args:
        segments: List of transcript segment dicts with 'start', 'end', 'text'
        start_time: Start of range in seconds
        end_time: End of range in seconds

    Returns:
        Newline-joined lines in the form "[12.3s-15.7s] text"
    """
    lines = []
    for seg in segments:
        if seg['end'] >= start_time and seg['start'] <= end_time:
            lines.append(
                f"[{seg['start']:.1f}s-{seg['end']:.1f}s] {seg.get('text', '')}"
            )
    return '\n'.join(lines)


def get_timestamped_words_for_range(
    segments: list[dict],
    start_time: float,
    end_time: float,
) -> str:
    """Get timestamped words from segments that provide word timing."""
    words = []
    for segment in segments:
        for word in segment.get('words') or []:
            try:
                start = float(word['start'])
                end = float(word.get('end', start))
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(start) or not math.isfinite(end) or end < start:
                continue
            if end >= start_time and start <= end_time:
                text = str(word.get('word', '')).strip()
                if text:
                    words.append(f"[{start:.2f}s-{end:.2f}s] {text}")
    return '\n'.join(words)


def extract_text_in_range(
    transcript: str,
    start: float,
    end: float,
    include_partial: bool = True
) -> str:
    """Extract text from VTT-formatted transcript within time range.

    Parses transcript in the format:
    [HH:MM:SS.mmm --> HH:MM:SS.mmm] Text content here

    Args:
        transcript: Full transcript text with timestamps
        start: Start time in seconds
        end: End time in seconds
        include_partial: If True, include segments that partially overlap
                        the range. If False, only include fully contained.

    Returns:
        Extracted text content, joined with spaces
    """
    return ' '.join(
        span['text']
        for span in extract_timed_spans_in_range(
            transcript, start, end, include_partial)
    )


def extract_timed_spans_in_range(
    transcript: str,
    start: float,
    end: float,
    include_partial: bool = True,
) -> list[dict]:
    """The timed spans extract_text_in_range joins, with their char offsets.

    Each dict is {'start', 'end', 'text', 'offset'}, where offset is the index
    of that span's text inside the joined string. Callers that need to map a
    character position in the extracted text back to a timestamp use this;
    extract_text_in_range delegates here so the two cannot drift apart.
    """
    if not transcript:
        return []

    # Pattern matches: [timestamp --> timestamp] text
    pattern = r'\[(\d{1,2}:\d{2}:\d{2}(?:\.\d{1,3})?)\s*-->\s*(\d{1,2}:\d{2}:\d{2}(?:\.\d{1,3})?)\]\s*([^\[]+)'

    spans: list[dict] = []
    offset = 0
    for match in re.finditer(pattern, transcript):
        seg_start = parse_timestamp(match.group(1))
        seg_end = parse_timestamp(match.group(2))
        text = match.group(3).strip()

        if not text:
            continue

        if include_partial:
            in_range = seg_end >= start and seg_start <= end
        else:
            in_range = seg_start >= start and seg_end <= end
        if not in_range:
            continue

        spans.append({'start': seg_start, 'end': seg_end,
                      'text': text, 'offset': offset})
        # +1 for the single space ' '.join inserts between spans.
        offset += len(text) + 1

    return spans


def extract_text_from_segments(
    segments: list[dict],
    start: float,
    end: float,
    max_words: int | None = None
) -> str:
    """Extract text from segment dicts within time range.

    Works with segment lists (dicts with 'start', 'end', 'text' keys)
    rather than VTT strings.

    Args:
        segments: List of segment dicts with start/end/text
        start: Start time in seconds
        end: End time in seconds
        max_words: Optional maximum word count limit

    Returns:
        Extracted text content, joined with spaces
    """
    spans = timed_spans_from_segments(segments, start, end)
    if not max_words:
        return ' '.join(span['text'] for span in spans)
    words: list[str] = []
    for span in spans:
        words.extend(span['text'].split())
        if len(words) >= max_words:
            break
    return ' '.join(words[:max_words])


def timed_spans_from_segments(
    segments: list[dict],
    start: float,
    end: float,
) -> list[dict]:
    """The spans extract_text_from_segments joins, with their char offsets.

    Segments-shaped sibling of extract_timed_spans_in_range, so callers holding
    segment dicts can map a character position back to a timestamp.
    extract_text_from_segments joins this, so the two cannot drift apart.
    """
    spans: list[dict] = []
    offset = 0
    for seg in segments:
        seg_start = seg.get('start', 0)
        seg_end = seg.get('end', 0)
        if seg_end < start or seg_start > end:
            continue
        text = (seg.get('text') or '').strip()
        if not text:
            continue
        spans.append({'start': seg_start, 'end': seg_end,
                      'text': text, 'offset': offset})
        offset += len(text) + 1
    return spans


def _segment_text_to_word_segments(seg: dict) -> list[dict] | None:
    """Convert a segment's text into word-level segments.
    Args:
        seg: A dictionary containing `text`, `start`, and `end` keys.
    Returns:
      A list of word-level segments with interpolated `start` and `end`
      times, or None if `seg` is invalid.
    """
    text = seg.get('text')
    if not isinstance(text, str):
        return None # bad input
    try:
        start = float(seg.get('start')) # type: ignore
        end = float(seg.get('end')) # type: ignore
    except (ValueError, TypeError):
        return None # bad input
    return _text_to_word_segments(text, start, end)

def _text_to_word_segments(text: str, start: float, end: float) -> list[dict]:
    """Convert `text` into word-level segments.
    Returns:
        A list of word-level segments with interpolated `start` and `end` times.
    """
    word_segments = []
    words = text.split() # splits on whitespace, omits empty
    word_start = start
    word_delta = (end - start) / len(words)
    for w in words:
        word_segments.append({
            'word': f" {w}",  # include leading space, even at start
            'start': round(word_start, 3),
            'end': round(word_start + word_delta, 3)
        })
        word_start += word_delta
    if word_segments: # preserve exact start/end
        word_segments[0]['start'] = start
        word_segments[-1]['end'] = end
    return word_segments

_END_SENTENCE_PUNCTUATION = ('.', '!', '?')
_END_QUOTES = ('"', '\u2019', '\u201d') # end quote chars

def _split_segment_into_sentences(seg: dict,
                                  do_not_interpolate: bool = False
                                  ) -> list[dict]:
    """Attempts to split the given segment into individual sentences based on
    clear punctuation. See `split_segments_into_sentences` for usage.
    """
    seg_words = seg.get('words')
    if not isinstance(seg_words, list):
        seg_words = None if do_not_interpolate else _segment_text_to_word_segments(seg)
    if not seg_words:
        return [seg] # cannot split into sentences without word segments

    sentences = []
    first_word, next_word = 0, 0
    while next_word < len(seg_words) - 1: # we handle the last word/sentence below
        word_seg = seg_words[next_word]
        word = (word_seg if isinstance(word_seg, dict) else {}).get('word')
        if not isinstance(word, str):
            return [seg] # bad input

        if (
            (len(word) >= 1 and word[-1] in _END_SENTENCE_PUNCTUATION) or
            (len(word) >= 2 and word[-1] in _END_QUOTES and word[-2] in _END_SENTENCE_PUNCTUATION)
        ):
            # found end of sentence
            sentence = seg_words[first_word:next_word+1]
            sentences.append({
                # each word includes leading space, if any (supports hyphenation)
                'text': ''.join(w['word'] for w in sentence).strip(),
                'start': sentence[0].get('start'),
                'end': sentence[-1].get('end'),
                'words': sentence
            })
            first_word = next_word + 1
        next_word += 1

    # handle the last word/sentence
    if first_word == 0:
        sentences.append(seg) # already a single sentence
    else:
        sentence = seg_words[first_word:]
        sentences.append({
            'text': ''.join(w['word'] for w in sentence).strip(),
            'start': sentence[0].get('start'),
            'end': sentence[-1].get('end'),
            'words': sentence
        })

    return sentences

def split_segments_into_sentences(segments: list[dict],
                                  do_not_interpolate: bool = False
                                  ) -> tuple[list[dict], int]:
    """Attempts to split the given segments into individual sentences based on
    clear punctuation. Each input segment is assumed to contain at least one
    complete sentence, thus segments will only be split and never merged.

    Args:
        segments (list[dict]): A list of segments, where each segment is a
          dictionary containing either a `words` key containing an ordered
          list of word-level segments, or a `text`, `start`, and `end` keys
          from which the words can be interpolated.
        do_not_interpolate (bool): If True, the function will not attempt to
          interpolate missing `words` from `text`, `start`, and `end` keys.

    Returns:
        tuple[list[dict], int]: A tuple containing a list of segments split into
          individual sentences and an integer indicating how many additional
          segments were created. If anything goes wrong, the original segments
          are returned unchanged.
    """
    sentence_segments = []
    for seg in segments:
        sentence_segments += _split_segment_into_sentences(
            seg, do_not_interpolate=do_not_interpolate,
        )
    return sentence_segments, len(sentence_segments) - len(segments)


def _segment_duration(seg: dict) -> float:
    return float(seg['end']) - float(seg['start'])

def _segment_gap(left: dict, right: dict) -> float:
    return float(right['start']) - float(left['end'])

def merge_segments(segments: list[dict],
                   maximum_gap: float = 0.5,
                   minimum_duration: float = 5.0,
                   maximum_duration: float = 30.0,
                   ) -> tuple[list[dict], int]:
    """Merge all segments with a duration less than `minimum_duration`
    with its shortest near neighbor. Segments will only be merged if
    the gap between them is less than `maximum_gap` and the merged
    segment does not exceed `maximum_duration`.

    This gives best results when the input segments are focused (e.g.
    sentences) and `maximum_gap` is less than the time between segments
    (~0.5-1.0 second) and more than the time between related sentences
    (~0.1-0.5 second).

    Args:
        segments: List of segment dicts with valid `start` and `end` keys
          and ordered by `start` and `end`.
        maximum_gap (float): The maximum allowed gap (in seconds) between
          segments to consider them for merging. Segments separated by a
          gap larger than this will not be merged.
        minimum_duration (float): Segments shorter than this duration will
          be merged with their shortest nearest neighbor.
        maximum_duration (float): The maximum length of merged segments
          (in seconds). If a merged segment would exceed this duration, it
          will not be merged.
    Returns:
        A list of segments and an integer indicating how many segments
        were merged.
    """
    original_len = len(segments)
    if len(segments) <= 1 or minimum_duration <= 0.0:
        return segments, 0

    # Algorithm: Merge shortest segments first:
    # 1. Find all short segments.
    # 2. Sort by shortest duration.
    # 3. Merge each short segment with shortest near neighbor.

    # use a "priority queue" ordered by shortest segment duration
    short_segments = []
    tie_breaker = count()
    def maybe_queue_short_segment(seg: dict) -> bool:
        duration = _segment_duration(seg)
        if duration < minimum_duration:
            # tie-breaker ensures that heapq never compares the segment dicts
            # when there are duplicate durations
            heapq.heappush(short_segments, (duration, next(tie_breaker), seg))
            return True
        return False

    # make a copy of all segments to avoid modifying the originals
    segments = [seg.copy() for seg in segments]
    for seg in segments:
        maybe_queue_short_segment(seg)

    while short_segments:
        # we cant remove merged segments from the heap but we can skip
        # them if the queued duration does not match the current duration
        queued_duration, _, shortest = heapq.heappop(short_segments)
        duration = _segment_duration(shortest)
        if duration != queued_duration:
            continue # outdated/merged since it was queued

        # try to merge...
        merged = None
        i = segments.index(shortest)
        left = segments[i - 1] if i > 0 else None
        right = segments[i + 1] if (i + 1) < len(segments) else None
        left_dur = _segment_duration(left) if left else math.inf
        right_dur = _segment_duration(right) if right else math.inf
        if (left
            and left_dur < right_dur # choose smaller neighbor
            and _segment_gap(left, shortest) < maximum_gap # must be near
            and left_dur + duration <= maximum_duration # must not exceed maximum duration
        ):
            # merge [left <-- shortest]
            left['end'] = shortest['end']
            if left.get('text') or shortest.get('text'):
                left['text'] = ' '.join([left.get('text', ''), shortest.get('text', '')])
            if left.get('words') or shortest.get('words'):
                left['words'] = left.get('words', []) + shortest.get('words', [])
            merged = left

        elif (right
              and _segment_gap(shortest, right) < maximum_gap # must be near
              and duration + _segment_duration(right) <= maximum_duration # must not exceed maximum duration
        ):
            # merge [shortest --> right]
            right['start'] = shortest['start']
            if right.get('text') or shortest.get('text'):
                right['text'] = ' '.join([shortest.get('text', ''), right.get('text', '')])
            if right.get('words') or shortest.get('words'):
                right['words'] = shortest.get('words', []) + right.get('words', [])
            merged = right

        if merged:
            segments.remove(shortest)
            maybe_queue_short_segment(merged)

    return segments, original_len - len(segments)
