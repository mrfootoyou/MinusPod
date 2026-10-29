---
name: podcast-segmenter
description: >-
  Segments podcast transcripts into structured JSON by identifying non-main
  content (intros, outros, recaps, teasers, transitions, interactions, ads,
  promotions) while retaining substantive content.
---

Identify all non-main content segments in a podcast transcript and output a compact JSON object.

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
7. **Output**: Return a valid JSON object matching the schema.

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
- **Mid-cue boundary quotes (`start_text`/`end_text`)**: If the segment begins or ends mid-cue, then quote the first 6 and last 6 words of the segment text. If there are fewer than 6 words, quote all available. A 'word' is strictly defined as an alphanumeric sequence (`[^\W_]+`). Ex: "10,000 B.C. is pre-history" = 7 words.
- **Promotional scope**: Include lead-in disclosures ("sponsored by..."), offers, URLs, codes, and disclaimers.
- **Intros & Cold Opens**: Substantive cold opens remain `main_content`. `intro` ends at guest introductions or substantive discussion; later previews are `teaser`.

## Sponsor Watchlists

Reference lists (non-exhaustive) to identify sponsors and resolve misheard transcript spellings:

### Common Sponsors
BetterHelp, Athletic Greens, AG1, Shopify, Amazon, Audible, Squarespace, HelloFresh, Factor, NordVPN, ExpressVPN, Mint Mobile, MasterClass, Calm, Headspace, ZipRecruiter, Indeed, LinkedIn Jobs, LinkedIn, Stamps.com, SimpliSafe, Ring, ADT, Casper, Helix Sleep, Purple, Brooklinen, Bombas, Manscaped, Dollar Shave Club, Harry's, Quip, Hims, Hers, Roman, Function of Beauty, Native, Liquid IV, Athletic Brewing, Magic Spoon, Thrive Market, Butcher Box, Blue Apron, DoorDash, Uber Eats, Grubhub, Instacart, Rocket Money, Credit Karma, SoFi, Acorns, Betterment, Wealthfront, PolicyGenius, Lemonade, State Farm, Progressive, Geico, Liberty Mutual, T-Mobile, Visible, FanDuel, DraftKings, BetMGM, Toyota, Hyundai, CarMax, Carvana, eBay Motors, ZocDoc, GoodRx, Care/of, Ritual, Seed, HubSpot, NetSuite, Monday.com, Notion, Canva, Grammarly, Babbel, Rosetta Stone, Blinkist, Raycon, Bose, MacPaw, CleanMyMac, Green Chef, Magic Mind, Honeylove, Cozy Earth, Quince, LMNT, Nutrafol, Aura, OneSkin, Incogni, Gametime, 1Password, Bitwarden, CacheFly, Deel, DeleteMe, Framer, Miro, Monarch Money, OutSystems, Spaceship, Thinkst Canary, ThreatLocker, Vanta, Veeam, Zapier, Zscaler, Capital One, Ford, WhatsApp

### Common Retail/Consumer Brands
Nordstrom, Macy's, Target, Walmart, Kohl's, Bloomingdale's, JCPenney, TJ Maxx, Home Depot, Lowe's, Best Buy, Costco, Gap, Old Navy, H&M, Zara, Nike, Adidas, Lululemon, Coach, Kate Spade, Michael Kors, Sephora, Ulta, Bath & Body Works, CVS, Walgreens, AutoZone, O'Reilly Auto Parts, Jiffy Lube, Midas, Gold Belly, Farmer's Dog, Caldera Lab, Monster Energy, Red Bull, Whole Foods, Trader Joe's, Kroger, GNC

<!-- {sponsor_database} -->

## Output Format

Return only a valid, compact JSON object with no Markdown wrappers or commentary:

{
  "segments": [
    {
      "start": number,
      "end": number,
      "start_text": string or null,
      "end_text": string or null,
      "category": "intro | outro | interaction | teaser | recap | transition | sponsor | cross_promo | self_promo",
      "reason": string,
      "sponsor_name": string or null,
      "confidence": number
    }
  ]
}

### Field Notes

- `start`, `end`: Full cue boundary timestamps or indices (no interpolation).
- `start_text`, `end_text`: Exact boundary text (up to 6 words) iif the segment begins or ends mid-cue; `null` otherwise.
- `reason`: Brief classification rationale.
- `sponsor_name`: Canonical brand/offering for promotional segments; `null` otherwise.
- `confidence`: Lowest-certainty component score (category, boundary, sponsor). Use `0.9+` only when alternatives are improbable, `0.7-0.8` when likely but ambiguous, and `≤0.6` when competing interpretations are plausible.

## Example: Mid-Cue Boundary

Input transcript:
```
[192.3-197.8] We'll get back to the interview in a moment. This episode is sponsored by Acme Coffee.
[198.0-200.6] Visit acme.example today to get 20% off your first order.
```

Output:
```json
{
  "segments": [
    {
      "start": 192.3,
      "end": 200.6,
      "start_text": "This episode is sponsored by Acme",
      "end_text": "get 20% off your first order.",
      "category": "sponsor",
      "reason": "Sustained audience-directed product promotion",
      "sponsor_name": "Acme Coffee",
      "confidence": 0.95
    }
  ]
}
```

Notice how `start` and `end` capture the full cue span, while `start_text` pinpoints the exact mid-cue boundary. `end_text` is non-null because `start_text` is non-null (they're a pair).