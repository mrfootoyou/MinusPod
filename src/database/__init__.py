"""SQLite database package for MinusPod."""
import os
import sqlite3
import threading
import time
import logging
from pathlib import Path

from database.schema import SchemaMixin
from database.podcasts import PodcastMixin
from database.episodes import EpisodeMixin
from database.settings import SettingsMixin
from database.patterns import PatternMixin
from database.sponsors import SponsorMixin
from database.stats import StatsMixin
from database.maintenance import MaintenanceMixin
from database.fingerprints import FingerprintMixin
from database.cue_templates import CueTemplateMixin
from database.cue_detections import CueDetectionMixin
from database.queue import QueueMixin
from database.search import SearchMixin
from database.auth_lockout import AuthLockoutMixin
from database.podping_hosts import PodpingHostMixin
from database.feed_subscribers import FeedSubscriberMixin
from database.downloads import DownloadHistoryMixin
from database.provider_admission import ProviderAdmissionMixin
from database.upload_reservations import UploadReservationMixin
from database.failover_events import FailoverEventsMixin
from database.pattern_cleanup import PatternCleanupMixin
from database.youtube import YouTubeMixin
from utils.paths import resolve_data_dir

logger = logging.getLogger(__name__)

# Slow statements and transactions are logged at this threshold.
# Implicit transaction timing includes the opening statement's lock wait.
SLOW_SQLITE_SECONDS = 5.0

# How long a writer waits for the lock. 30s left almost no margin: ordinary
# feed-refresh writes were observed holding it 28.4s under contention.
BUSY_TIMEOUT_MS = 60000
_sqlite_metrics_lock = threading.Lock()
_sqlite_metrics = {
    'slowStatements': 0,
    'slowCommits': 0,
    'failedCommits': 0,
    'longTransactions': 0,
    'lastCommitMs': 0.0,
    'maxCommitMs': 0.0,
}


def sqlite_metrics_snapshot() -> dict:
    with _sqlite_metrics_lock:
        return {'scope': 'worker_process', 'processId': os.getpid(), **_sqlite_metrics}


def _record_sqlite_metric(name: str, value: float | None = None) -> None:
    with _sqlite_metrics_lock:
        if value is None:
            _sqlite_metrics[name] += 1
        else:
            _sqlite_metrics['lastCommitMs'] = value
            _sqlite_metrics['maxCommitMs'] = max(_sqlite_metrics['maxCommitMs'], value)
            if value >= SLOW_SQLITE_SECONDS * 1000:
                _sqlite_metrics['slowCommits'] += 1


class TracedConnection(sqlite3.Connection):
    """sqlite3.Connection that logs slow statements and long transactions."""

    _tx_started = None
    _tx_opener = None
    _tx_start_includes_wait = False

    def execute(self, sql, *args):
        return self._traced(super().execute, sql, *args)

    def executemany(self, sql, *args):
        return self._traced(super().executemany, sql, *args)

    def _traced(self, run, sql, *args):
        was_in_tx = self.in_transaction
        started = time.monotonic()
        try:
            return run(sql, *args)
        except Exception:
            # A statement that opened the transaction and then failed would
            # leave it open on this thread with nothing to roll it back (#566).
            if not was_in_tx and self.in_transaction:
                super().rollback()
            raise
        finally:
            self._note_statement(sql, started, was_in_tx)

    def commit(self):
        tx_started = self._tx_started
        tx_opener = self._tx_opener
        started = time.monotonic()
        try:
            super().commit()
        except Exception:
            _record_sqlite_metric('failedCommits')
            raise
        else:
            _record_sqlite_metric('commit', (time.monotonic() - started) * 1000)
            self._note_transaction_end('commit', tx_started, tx_opener)

    def rollback(self):
        tx_started = self._tx_started
        tx_opener = self._tx_opener
        try:
            super().rollback()
        finally:
            self._note_transaction_end('rollback', tx_started, tx_opener)

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            try:
                self.commit()
            except BaseException:
                try:
                    self.rollback()
                except Exception:
                    pass
                raise
        else:
            self.rollback()
        return False

    def _note_statement(self, sql, started, was_in_tx):
        elapsed = time.monotonic() - started
        if elapsed >= SLOW_SQLITE_SECONDS:
            _record_sqlite_metric('slowStatements')
            logger.warning(
                "SQLite statement took %.1fs on thread %s: %s",
                elapsed, threading.current_thread().name, _sql_head(sql))
        if not self.in_transaction:
            # Autocommit, or a C-level commit (`with conn:`, executescript) ended it.
            self._tx_started = self._tx_opener = None
            self._tx_start_includes_wait = False
        elif not was_in_tx:
            # BEGIN IMMEDIATE may spend most of its time waiting for the lock.
            normalized = str(sql).lstrip().upper()
            immediate = normalized.startswith('BEGIN IMMEDIATE')
            self._tx_started = time.monotonic() if immediate else started
            self._tx_start_includes_wait = not immediate
            self._tx_opener = _sql_head(sql)
        elif self._tx_opener and self._tx_opener.split(None, 1)[0].upper() == 'BEGIN':
            # Name the first real statement instead of the BEGIN that opened it.
            self._tx_opener = _sql_head(sql)

    def _note_transaction_end(self, how, tx_started=None, tx_opener=None):
        if tx_started is None:
            return
        held = time.monotonic() - tx_started
        if held >= SLOW_SQLITE_SECONDS:
            _record_sqlite_metric('longTransactions')
            if self._tx_start_includes_wait:
                logger.warning(
                    "SQLite transaction elapsed %.1fs (opening statement and possible lock wait included) before %s on thread %s; opened by: %s",
                    held, how, threading.current_thread().name, tx_opener)
            else:
                logger.warning(
                    "SQLite write transaction held %.1fs before %s on thread %s; opened by: %s",
                    held, how, threading.current_thread().name, tx_opener)
        self._tx_started = None
        self._tx_opener = None
        self._tx_start_includes_wait = False


def _sql_head(sql):
    return truncate(' '.join(str(sql).split()), 120)

from utils.constants import DEFAULT_SYSTEM_PROMPT  # re-exported for backward compat
from utils.text import truncate

# Verification pass prompt - runs on processed audio to catch missed ads
DEFAULT_VERIFICATION_PROMPT = """
You are a JSON API for segmenting podcast transcripts.

## Task

Identify all non-main content segments in a podcast transcript and output a compact JSON object.

<!-- BEGIN Verification Customization -->
Note: This is the **second pass** over the transcript. Ads and other unwanted segments have already been cut. However, the first-pass may have missed some segments, in-whole or in-part. Be especially vigilant for "orphaned fragments" — short segments that appear disconnected from the surrounding context. Examples include:
- Out-of-context URLs, promo codes, calls to action
- Out-of-place sponsor mentions: "that's [brand].com", "thanks to [sponsor]"
- Ad bumpers and in/out cues that survived the first pass.
<!-- END Verification Customization -->

Input transcript use one of two cue formats:
- **Timestamped**: `[start_time - end_time] Text` (seconds).
- **Indexed**: `[index] Text` (zero-based).

Metadata (show ID, host, guest, date, sponsors, chapters) may be provided to help resolve transcription errors and boundary hints; always verify against the transcript itself.

## Procedure

1. **Ingest cues & metadata**: Read sequential cues (timestamped or indexed).
2. **Detect boundaries**: Identify functional/topic shifts (may occur mid-cue).
3. **Classify**: Assign segments to categories; default ambiguous spans to `main_content`.
4. **Resolve promotions**: Identify beneficiary (`sponsor`, `cross_promo`, `self_promo`) and correct names using metadata and watchlists.
5. **Consolidate**: Merge adjacent same-category non-promotional segments < ~5s. Keep promotional segments separate even if contiguous.
6. **Filter**: Omit `main_content` from output; callers reconstruct it from gaps.
7. **Extract boundary text**: Quote up to 6 words (~30 characters) marking exact segment edges within the boundary cues.
8. **Output**: Return a valid JSON object matching the schema.

<!-- BEGIN Verification Customization -->
**Note**: Orphaned fragments may be especially difficult to classify accurately. Do your best and know that correctly identified orphaned fragments will be merged with their parent segment, making classification less critical.

Orphaned fragments are also exempt from the minimum duration consolidation rule since they may be short by their very nature.
<!-- END Verification Customization -->

## Categories

- `main_content`: Substantive episode material (stories, reporting, interviews, discussions, Q&A, cold opens, current-episode summaries, guest introductions and bios). Never returned in output.
- `intro`: Non-substantive opening material (theme music, show ID, boilerplate host intro, opening previews). May follow a cold open. Ends when substantive discussion or guest introduction begins.
- `outro`: Explicit episode wrap-up following substantive discussion (sign-off, credits, closing theme, network tagline). Post-credit discussions remain `main_content`.
- `interaction`: Direct calls for audience participation (reviews, comments, question submissions). Merely addressing listeners does not qualify.
- `teaser`: Mid-episode preview outside the intro pointing to content later in the current episode.
- `recap`: Summary of prior episodes ("previously on"). Current-episode summaries remain `main_content`.
- `transition`: Significant musical or spoken interlude bridging sections without substantive content.
- `sponsor`: Sustained promotion for an outside commercial brand, product, service, or PSA.
- `cross_promo`: Sustained promotion for a sibling show, channel, event, or network bundle.
- `self_promo`: Sustained promotion for the current show, host, memberships, merchandise, or show channels.

**Default to retention (`main_content`)**: When in doubt, classify spans as `main_content`. Mark non-main content only if listeners can safely discard it without losing context, essential information, or conversational flow. Passing brand mentions, neutral commentary, and guest plugs remain `main_content`.

## Boundary & Clipping Rules

- **Conservative clipping**: Bound non-main segments narrowly. When edges are blurry, err toward keeping audio in `main_content`.
- **Cue spans (`start`/`end`)**: Must cover the full cue span containing the segment. Use the first cue's start time/index and the last cue's end time/index. Do not interpolate.
- **Mid-cue overlap**: If a cue contains boundaries between multiple segments, include that cue in the span of every segment it overlaps.
- **Boundary quotes (`start_text`/`end_text`)**: Quote up to 6 words (~30 characters) marking the exact segment edge inside the boundary cue. Use `""` if the cue text is empty.
- **Promotional scope**: Include lead-in disclosures ("sponsored by..."), offers, URLs, codes, and disclaimers.
- **Intros & Cold Opens**: Substantive cold opens remain `main_content`. `intro` ends at guest introductions or substantive discussion; later previews are `teaser`.

<!-- BEGIN Verification Customization -->
<!-- No sponsor watchlists
## Sponsor Watchlists

Reference lists (non-exhaustive) to identify sponsors and resolve misheard transcript spellings:

### Common Sponsors
BetterHelp, Athletic Greens, AG1, Shopify, Amazon, Audible, Squarespace, HelloFresh, Factor, NordVPN, ExpressVPN, Mint Mobile, MasterClass, Calm, Headspace, ZipRecruiter, Indeed, LinkedIn Jobs, LinkedIn, Stamps.com, SimpliSafe, Ring, ADT, Casper, Helix Sleep, Purple, Brooklinen, Bombas, Manscaped, Dollar Shave Club, Harry's, Quip, Hims, Hers, Roman, Function of Beauty, Native, Liquid IV, Athletic Brewing, Magic Spoon, Thrive Market, Butcher Box, Blue Apron, DoorDash, Uber Eats, Grubhub, Instacart, Rocket Money, Credit Karma, SoFi, Acorns, Betterment, Wealthfront, PolicyGenius, Lemonade, State Farm, Progressive, Geico, Liberty Mutual, T-Mobile, Visible, FanDuel, DraftKings, BetMGM, Toyota, Hyundai, CarMax, Carvana, eBay Motors, ZocDoc, GoodRx, Care/of, Ritual, Seed, HubSpot, NetSuite, Monday.com, Notion, Canva, Grammarly, Babbel, Rosetta Stone, Blinkist, Raycon, Bose, MacPaw, CleanMyMac, Green Chef, Magic Mind, Honeylove, Cozy Earth, Quince, LMNT, Nutrafol, Aura, OneSkin, Incogni, Gametime, 1Password, Bitwarden, CacheFly, Deel, DeleteMe, Framer, Miro, Monarch Money, OutSystems, Spaceship, Thinkst Canary, ThreatLocker, Vanta, Veeam, Zapier, Zscaler, Capital One, Ford, WhatsApp

### Common Retail/Consumer Brands
Nordstrom, Macy's, Target, Walmart, Kohl's, Bloomingdale's, JCPenney, TJ Maxx, Home Depot, Lowe's, Best Buy, Costco, Gap, Old Navy, H&M, Zara, Nike, Adidas, Lululemon, Coach, Kate Spade, Michael Kors, Sephora, Ulta, Bath & Body Works, CVS, Walgreens, AutoZone, O'Reilly Auto Parts, Jiffy Lube, Midas, Gold Belly, Farmer's Dog, Caldera Lab, Monster Energy, Red Bull, Whole Foods, Trader Joe's, Kroger, GNC
-->
<!-- END Verification Customization -->

{sponsor_database}

## Output Format

Return only a valid, compact JSON object with no Markdown wrappers or commentary:

{
  "segments": [
    {
      "start": number,
      "end": number,
      "start_text": string,
      "end_text": string,
      "category": "intro | outro | interaction | teaser | recap | transition | sponsor | cross_promo | self_promo",
      "reason": string,
      "sponsor_name": string or null,
      "confidence": number
    }
  ]
}

### Field Notes

- `start`, `end`: Full cue boundary timestamps or indices (no interpolation).
- `start_text`, `end_text`: Exact boundary text (≤6 words, ~30 chars).
- `reason`: Brief classification rationale.
- `sponsor_name`: Canonical brand/offering for promotional segments; `null` otherwise.
- `confidence`: Lowest-certainty component score (category, boundary, sponsor). Use `0.9+` only when alternatives are improbable, `0.7-0.8` when likely but ambiguous, and `≤0.6` when competing interpretations are plausible.

## Example: Mid-Cue Boundary

Input transcript:
```
[192.3-194.8] We'll get back to the interview in a moment.
[195.0-198.6] Visit Acme dot com to get started today. Now, Maya, what did the team discover?
```

Output:
```json
{
  "segments": [
    {
      "start": 195.0,
      "end": 198.6,
      "start_text": "Visit Acme dot com to get started",
      "end_text": "Acme dot com to get started today.",
      "category": "sponsor",
      "reason": "Appears to be a leftover ad fragment",
      "sponsor_name": "Acme",
      "confidence": 0.85
    }
  ]
}
```

Notice how `start` and `end` capture the full cue span, while `start_text` and `end_text` pinpoint text boundaries within the start and end cues. The texts overlap in this example because the ad fragment is short. Confidence is on the low end because it appears to be a fragment.
""".strip()


# Both reviewer prompts use placeholder substitution via _render_prompt;
# never .format() these strings directly (the JSON examples contain literal
# curly braces that .format() would attempt to interpolate).
DEFAULT_REVIEW_PROMPT = """You are reviewing a candidate advertisement that has already been detected in a podcast episode. The transcript shows the candidate ad clearly marked, with up to 60 seconds of context before and after.

Your job is to return the corrected ad segment, OR an empty array if it is not actually an ad. Treat this exactly like ad detection on a single short window: you are emitting the ad object that should be cut from the audio, with start and end timestamps, or no object at all.

KEEP THE AD (return one segment): The candidate is a real-world advertisement that should be cut. Use the original start and end if they are already correct. Adjust them when the boundaries clip into show content or miss part of the ad:
- Start should land at or just before the first promotional word or transition phrase ("let's take a break", "and now a word from", "this episode is brought to you by")
- End should land at or just after the last call to action (final URL, promo code, sign-off), not in the middle of show content that follows
- Read adjusted timestamps off the [start-end] stamps on the transcript lines themselves, including the context lines outside the candidate markers. Never interpolate a boundary
- Adjusted boundaries must stay within {max_boundary_shift_seconds} seconds of the original boundaries in either direction

PARTIAL SPAN: if any part of the candidate span is not ad content (show content before the ad starts or after it ends), you MUST return adjusted start and end timestamps covering only the ad portion. Never return the original boundaries and describe the trim only in the "reason" text: the reason is prose for a human, and only the start and end numbers control the cut.

DROP THE AD (return empty array): The candidate is not a real-world advertisement. Reject cases:
- A guest discussing their own work, book, or project in the context of the interview
- The host organically mentioning their own other shows, social media, or Patreon as part of conversational flow (not a produced segment)
- Brand names mentioned in passing as part of genuine topic discussion, news coverage, or product reviews
- A comedic bit or fictional sponsor read that is part of the show's creative content (no real product behind it)
- Silence, pauses, or audio production artifacts with no promotional transcript content
- Topic transitions or content gaps without promotional language
You may also return [{"is_ad": false, "reason": "why"}] instead of an empty array to reject explicitly.

DO NOT REJECT (these ARE real ads, keep them):
- Host-read sponsor segments, including ones without promo codes
- Hosting platform pre/post-rolls (Acast, Spotify for Podcasters, iHeart Radio, etc.)
- Cross-promotions for other podcasts inserted by the platform or network (different host or voice, different topic, sounds produced)
- Network promos
- Short brand tagline ads (15-45 seconds) that sound like polished radio commercials, even without promo codes or URLs
- Dynamically inserted retail or consumer brand ads

The distinction between editorial mention and ad: an ad is paid promotional content with a sponsor name and a value proposition aimed at the listener. Editorial discussion is the host or guest talking about a topic, even if a brand name comes up.

AUDIO CUE SIGNALS: if the prompt notes a labelled audio cue next to the candidate boundary, treat it as ground truth for that side and do not move the boundary across it. When a "cue_pair" candidate is shown, the matcher bracketed the break with two cues but the transcript may be sparse; keep the ad if any promotional language sits between the cues, even if the boundaries look loose.

Protected audio lines are hard limits your boundaries must not cross, and fingerprint spans are projected lengths, so prefer the transcript's precise edges.

A line marked as audio with no transcript inside the candidate may be speech the transcript missed; do not move an edge across it.

WHEN IN DOUBT: Keep the ad with original boundaries unchanged. Do not drop unless you have clear evidence from the transcript that the segment is not a real-world advertisement. Do not adjust unless the boundary error is unambiguous from the surrounding context. The cost of leaving a real ad in the audio (false negative) is higher than the cost of keeping a borderline detection.

OUTPUT FORMAT:
Return ONLY a valid JSON array. No explanation, no markdown.

Each kept ad: {"is_ad": BOOLEAN_TRUE_ONLY_IF_REAL_AD, "start": FLOAT_SECONDS, "end": FLOAT_SECONDS, "confidence": FLOAT_0_TO_1, "reason": "brief description"}

Set "is_ad" true only when the span is a real advertisement to cut. Set it false, or return an empty array, when it is not.

ALL values for "start", "end", and "confidence" MUST be numeric (float). Never use strings like "high", "low", "medium", or percentages like "95%". Examples: "start": 45.0, "end": 82.0, "confidence": 0.95

EXAMPLE - KEEP UNCHANGED (boundaries are correct):
Original boundaries: 1245.00s - 1320.50s.
[1240.0-1245.0] We will pick that thread back up in a minute.
>>> CANDIDATE AD START [1245.0s] >>>
[1245.0-1248.0] This episode is brought to you by BetterHelp.
[1248.0-1315.0] BetterHelp is the largest online therapy platform...
[1315.0-1320.5] Visit betterhelp.com slash podcast. That's betterhelp.com slash podcast.
<<< CANDIDATE AD END [1320.5s] <<<
[1320.5-1325.0] Anyway, back to what we were talking about.

Output: [{"is_ad": true, "start": 1245.0, "end": 1320.5, "confidence": 0.95, "reason": "Confirmed BetterHelp host-read sponsor with clean boundaries"}]

EXAMPLE - ADJUST BOUNDARIES (start was late, end was early):
Original boundaries: 100.00s - 130.00s.
[92.0-95.0] So that wraps up our discussion. Let's take a quick break.
[95.0-100.0] This episode is brought to you by Athletic Greens.
>>> CANDIDATE AD START [100.0s] >>>
[100.0-128.0] AG1 is the daily foundational nutrition supplement...
[128.0-130.0] Go to athleticgreens.com slash podcast.
<<< CANDIDATE AD END [130.0s] <<<
[130.0-132.0] That's athleticgreens.com slash podcast.
[132.0-135.0] Now, back to our conversation.

Both adjusted timestamps are read straight off context lines: 95.0 starts the line before the marker, 132.0 ends the line after it.

Output: [{"is_ad": true, "start": 95.0, "end": 132.0, "confidence": 0.92, "reason": "Started at the transition line at 95.0s; ended at 132.0s after the repeated URL"}]

EXAMPLE - DROP (host mentioning a brand editorially, not an ad):
Original boundaries: 50.00s - 70.00s.
[45.0-50.0] Have you been following the Apple antitrust case?
>>> CANDIDATE AD START [50.0s] >>>
[50.0-62.0] The DOJ argued that Apple's app store policies harm developers.
[62.0-70.0] They want the court to force a change to the commission structure.
<<< CANDIDATE AD END [70.0s] <<<
[70.0-74.0] What's your take on the proposed remedies?

Output: []{sponsor_database}"""


DEFAULT_RESURRECT_PROMPT = """You are taking a second look at a segment that the validator already rejected for low confidence. The transcript shows the candidate clearly marked, with up to 60 seconds of context before and after.

Your job: if the transcript shows this is actually an ad that should be cut, return the ad segment. If the validator was right and it is not an ad, return an empty array.

RESURRECT (return one segment): The validator was wrong. The segment is a real-world advertisement and should be cut. Resurrect when:
- The transcript clearly contains promotional language: sponsor name + value proposition + call to action, or polished marketing copy with concentrated brand messaging
- The segment matches the structure of an ad (transition in, sponsor read or platform promo, return to content) even if the validator marked confidence as low
- It is a short brand tagline ad (15-45 seconds) without promo codes, but with concentrated marketing language
- It is a hosting-platform pre/post-roll or cross-promo for another podcast in the network

KEEP REJECTED (return empty array): Agree with the validator. The segment is not a real-world advertisement. Match the same not-an-ad criteria the main reviewer uses:
- A guest discussing their own work in the context of the interview
- Host organically mentioning their own other shows or social media in conversation
- Brand names mentioned in passing as part of editorial topic discussion
- Comedic bits or fictional sponsor reads in the show's creative content
- Silence, pauses, or topic transitions with no promotional transcript content
You may also return [{"is_ad": false, "reason": "why"}] instead of an empty array to reject explicitly.

AUDIO CUE SIGNALS: if a labelled audio cue brackets or sits next to the rejected segment, treat it as strong evidence a break happened there; a cue-bracketed segment with even a single sponsor mention is usually a real ad the validator was too cautious about -- resurrect it.

WHEN IN DOUBT: Agree with the validator and return empty. Only resurrect when the transcript shows clear evidence that this is a real ad. The validator already saw reason to flag low confidence; do not override without evidence.

OUTPUT FORMAT:
Return ONLY a valid JSON array. No explanation, no markdown.

Each resurrected ad: {"is_ad": BOOLEAN_TRUE_ONLY_IF_REAL_AD, "start": FLOAT_SECONDS, "end": FLOAT_SECONDS, "confidence": FLOAT_0_TO_1, "reason": "brief description"}

Set "is_ad" true only when the span is a real advertisement to cut. Set it false, or return an empty array, when it is not.

ALL values for "start", "end", and "confidence" MUST be numeric (float). Never use strings like "high", "low", "medium", or percentages like "95%". Examples: "start": 45.0, "end": 82.0, "confidence": 0.95

EXAMPLE - RESURRECT (validator missed a real ad):
Validator-rejected segment: 666.7s - 674.3s (validator confidence 0.71)
[660.0s] So that's our take on the antitrust case.
[666.7s] Hosted on Acast. See acast dot com slash privacy for more information.
[674.5s] Welcome back, today we're talking about the Switch 2 launch.

Output: [{"is_ad": true, "start": 666.7, "end": 674.3, "confidence": 0.92, "reason": "Acast hosting platform post-roll, clearly promotional and not editorial"}]

EXAMPLE - KEEP REJECTED (validator was right):
Validator-rejected segment: 200.0s - 215.0s (validator confidence 0.65)
[195.0s] We've been talking about Apple's new privacy framework.
[200.0s] Apple says the new framework gives users more control over data sharing.
[215.0s] But critics argue Apple still has too much power over the app store.
[220.0s] Let's get into the developer reaction next.

Output: []{sponsor_database}"""


# Chapter topic-detection prompt. Rendered via utils.prompt.render_prompt with
# the block placeholders below; must stay byte-identical to the pre-setting
# f-string output when unset.
DEFAULT_CHAPTER_PROMPT = """Analyze this podcast transcript segment and identify {num_splits} major topic changes.

The segment runs from {segment_start} to {segment_end}.

For each topic change, provide the timestamp (from the [MM:SS] markers) and a short title (3-7 words).

OUTPUT FORMAT:
Return ONLY topic lines, one per line. No introduction, no explanation, no numbering.
Each line must be exactly: MM:SS Topic Title Here

Example:
05:30 Discussion of AI Trends
12:45 New Product Announcements

Only include clear topic transitions, not minor tangents. Skip the very beginning since that's already a chapter.{continuation_block}{description_block}{hints_block}

Transcript:
{transcript}"""


# Pattern cleanup system prompt (Experiments > Pattern cleanup). The user
# message carries pattern text, sponsor, category, and optional transcript context.
DEFAULT_PATTERN_CLEANUP_PROMPT = """You review one learned podcast segment pattern. Its text may include unrelated words before or after the intended segment.

Classify the intended segment first, then check its content.
- Preserve content that belongs to an intro, outro, recap, self-promotion, or listener interaction. Those categories are not contamination by themselves.
- For non-sponsor segments, use action "keep" with a category correction when needed. Do not invent a sponsor or force the segment into sponsor-copy trimming rules.
- For sponsor reads, retain only the exact sponsor copy. Drop host banter, unrelated show content, and transitions into or out of the break.
- If two or more sponsors are read back to back, return each read as a separate piece.
- Name the sponsor as it is spoken in the read. For a trim, also return sponsor when the recorded sponsor is wrong and the corrected name appears in the kept text.
- Return category when the recorded category is wrong. Use only sponsor, cross_promo, self_promo, interaction, intro, outro, or recap.
- Do not invent, reword, or reorder words. Every text you return must be copied from the pattern text.
- If the pattern is already clean, return action "keep".
- Set contaminated to true when the pattern holds unrelated main-show discussion that a trim cannot fix, and say why in contamination_reason.

Transcript context, when given, shows the audio around the pattern with the pattern marked between [[ and ]]. Use it only to judge where the intended segment starts and ends; never copy words from outside the pattern text.

Return one JSON object:
{"action": "keep" | "trim" | "split" | "rename", "text": string or null (trim only), "sponsor": string or null (rename, or optional with trim), "category": string or null (optional category correction), "pieces": [{"text": string, "sponsor": string, "category": string or null}] (split only; category optional per piece), "contaminated": boolean, "contamination_reason": string or null, "confidence": number from 0 to 1, "reasons": [short strings]}"""


class Database(SchemaMixin, PodcastMixin, EpisodeMixin, SettingsMixin,
               PatternMixin, SponsorMixin, StatsMixin, MaintenanceMixin,
               FingerprintMixin, CueTemplateMixin, CueDetectionMixin,
               QueueMixin, SearchMixin, AuthLockoutMixin, PodpingHostMixin,
               FeedSubscriberMixin, ProviderAdmissionMixin,
               UploadReservationMixin, FailoverEventsMixin, PatternCleanupMixin,
               DownloadHistoryMixin, YouTubeMixin):
    """SQLite database manager with thread-safe connections."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, data_dir: str = None):
        """Singleton pattern for database instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, data_dir: str = None):
        if self._initialized:
            return

        # Resolved at call time, not from an import-time constant: a bare
        # Database() after the singleton is reset must land in the configured
        # data dir, not the packaged default.
        self.data_dir = Path(data_dir or resolve_data_dir())
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "podcast.db"
        self._local = threading.local()
        self._initialized = True

        # Initialize schema
        self._init_schema()

        # Run migration if needed
        self._migrate_from_json()

        try:
            conn = self.get_connection()
            ap_count = conn.execute(
                "SELECT COUNT(*) FROM ad_patterns WHERE is_active = 1"
            ).fetchone()[0]
            ks_count = conn.execute(
                "SELECT COUNT(*) FROM known_sponsors WHERE is_active = 1"
            ).fetchone()[0]
            logger.debug(
                f"Pattern catalog: ad_patterns active={ap_count}, "
                f"known_sponsors active={ks_count}"
            )
        except Exception as e:
            logger.warning(f"Pattern catalog count failed: {e}")

    def get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection.

        WAL mode is preferred but not required: a stale or permissioned
        WAL file on the mounted volume occasionally makes the first
        ``PRAGMA journal_mode = WAL`` fail with "disk I/O error". When
        that happens, reset the journal via ``PRAGMA journal_mode = DELETE``
        and try WAL once more. If WAL is still refused, stay on DELETE
        mode -- less concurrent, still correct -- rather than crash the
        worker on boot.
        """
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=BUSY_TIMEOUT_MS / 1000,
                factory=TracedConnection,
            )
            self._local.connection.row_factory = sqlite3.Row
            self._local.connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
            self._local.connection.execute("PRAGMA foreign_keys = ON")
            try:
                self._local.connection.execute("PRAGMA journal_mode = WAL")
            except sqlite3.OperationalError as exc:
                logger.warning(
                    "PRAGMA journal_mode = WAL failed (%s); resetting WAL "
                    "state and retrying.",
                    exc,
                )
                try:
                    self._local.connection.execute("PRAGMA journal_mode = DELETE")
                    self._local.connection.execute("PRAGMA journal_mode = WAL")
                except sqlite3.OperationalError:
                    logger.warning(
                        "WAL mode still refused after reset; falling back "
                        "to DELETE journal. Concurrency is reduced until "
                        "the volume state is repaired.",
                    )
            # NORMAL sync gives WAL its durability contract (fsync on
            # checkpoint and WAL commit) without the fsync-every-write
            # penalty of FULL. Harmless in DELETE mode too.
            try:
                self._local.connection.execute("PRAGMA synchronous = NORMAL")
                self._local.connection.execute("PRAGMA wal_autocheckpoint = 1000")
            except sqlite3.OperationalError:
                pass
        return self._local.connection

    class _TransactionContext:
        """Context manager for database transactions with automatic commit/rollback."""
        def __init__(self, conn, immediate=False):
            self.conn = conn
            self.immediate = immediate
        def __enter__(self):
            if self.immediate:
                if self.conn.in_transaction:
                    # BEGIN IMMEDIATE cannot nest; a transaction open here is a leak.
                    logger.warning("Rolled back a leaked transaction before BEGIN IMMEDIATE on thread %s",
                                   threading.current_thread().name)
                    self.conn.rollback()
                self.conn.execute("BEGIN IMMEDIATE")
            return self.conn
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
            return False

    def transaction(self, immediate: bool = False):
        """Context manager for database transactions.

        Usage:
            with db.transaction() as conn:
                conn.execute("INSERT ...")
                conn.execute("UPDATE ...")
            # Auto-commits on success, auto-rolls back on exception

        immediate=True issues BEGIN IMMEDIATE so the write lock is taken
        up front, where busy_timeout applies. The default deferred begin
        upgrades to a write lock at the first DML statement, and that
        upgrade fails instantly with "database is locked" when the
        snapshot is stale (SQLITE_BUSY_SNAPSHOT is not retried by
        busy_timeout). Use it for multi-statement writes that can run
        alongside other writers (issue #566).

        Do not hold one across per-row Python work: a transaction spanning a
        whole feed's episodes held the write lock past every other writer's
        busy_timeout. Chunk the rows and take a transaction per chunk.
        """
        return self._TransactionContext(self.get_connection(), immediate=immediate)

    def rollback_open_transaction(self) -> bool:
        """Roll back a transaction left open on this thread's connection.

        Returns True when there was one. Connections are thread-local and
        live for the worker thread's lifetime, so a write path that raises
        between BEGIN and commit without rolling back would otherwise hold
        its lock forever and fail every later write on the thread with
        "database is locked" (issue #566). Does not create a connection.
        """
        conn = getattr(self._local, 'connection', None)
        if conn is not None and conn.in_transaction:
            conn.rollback()
            return True
        return False

    def clear_leaked_transaction(self, log, where: str) -> None:
        """Logged, never-raising rollback_open_transaction. Called from the
        guard points that bound a leaked transaction's lifetime: request
        teardown, background loop iterations, and the episode processing
        thread."""
        try:
            if self.rollback_open_transaction():
                log.warning(f"Rolled back a leaked transaction ({where})")
        except Exception as e:
            log.error(f"Leaked-transaction rollback failed ({where}): {e}")
