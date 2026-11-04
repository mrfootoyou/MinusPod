"""Prompt placeholder substitution helpers.

Detection, verification, and reviewer prompts all use ``{placeholder}``
substitution rather than appending content unconditionally. ``str.replace``
(not ``str.format``) is intentional so literal ``{...}`` JSON examples in
prompt bodies stay intact (also avoids injection attacks).
"""
import re

from config import DETECTION_NOTES_MAX_CHARS

SPONSOR_DATABASE_HEADER = (
    "\n\nDYNAMIC SPONSOR DATABASE (current known sponsors - treat as high confidence):\n"
)


def render_prompt(prompt: str, **vars: str) -> str:
    """Substitute ``{name}`` placeholders in ``prompt`` with provided values.

    Variables without a corresponding placeholder are silently dropped: that
    is the supported way for a user to opt out of an injection by removing
    the placeholder from their customized prompt.
    """
    rendered = prompt
    for name, value in vars.items():
        rendered = rendered.replace('{' + name + '}', value)
    return rendered


def render_prompt_once(prompt: str, **vars: str) -> str:
    """Single-pass ``{name}`` substitution over the template only.

    Required when values carry untrusted text (transcripts, feed
    descriptions): a substituted value is never rescanned, so it cannot
    smuggle another placeholder into the prompt.
    """
    if not vars:
        return prompt
    pattern = re.compile('|'.join(r'\{' + re.escape(k) + r'\}' for k in vars))
    return pattern.sub(lambda m: vars[m.group(0)[1:-1]], prompt)


def format_sponsor_block(sponsor_list: str) -> str:
    """Wrap a non-empty sponsor list with the standard header.

    Empty list returns empty string so substitution does not produce a
    dangling header on prompts whose ``{sponsor_database}`` placeholder is
    left in place.
    """
    if not sponsor_list:
        return ""
    return SPONSOR_DATABASE_HEADER + sponsor_list


OVERRIDE_HEADER = "\n\nADDITIONAL INSTRUCTIONS (these take precedence):\n"


def apply_override(prompt: str, override: str) -> str:
    """Inject an optional per-pass override into an already-rendered prompt.

    Empty/None override leaves the prompt unchanged, so the built-in default
    prompts render byte-identically to today. When the prompt contains an
    ``{override}`` placeholder, the user's text is inserted there verbatim -- they
    control its placement and the wording around it. Otherwise the override is
    appended under a precedence header.
    """
    if not override or not override.strip():
        return prompt.replace('{override}', '') if '{override}' in prompt else prompt
    if '{override}' in prompt:
        return prompt.replace('{override}', override)
    return prompt + OVERRIDE_HEADER + override

def strip_html(text: str|None) -> str:
    """Convert simple HTML to plain text for show-note timestamp parsing.

    Block-level tags must be turned into newlines (not just stripped) so the
    downstream `_TIMESTAMP_PATTERNS` regex sees each timestamp on its own line.
    A bare tag-stripper like nh3 would collapse `<p>00:00 A</p><p>05:30 B</p>`
    into `00:00 A05:30 B` and miss every anchor after the first.
    """
    if not text:
        return ""
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</(p|li|div)>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    for entity, char in (('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'),
                         ('&quot;', '"'), ('&#39;', "'"), ('&nbsp;', ' ')):
        text = text.replace(entity, char)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def scrub_description(description: str|None, max_length: int|None = None, split_location: float = 1.0) -> str:
    """Scrub the description of HTML, timestamps, URLs, excessive whitespace,
    and then truncate to `max_length` characters by splitting on a word
    boundary at the specified location and inserting ellipsis.

    Args:
        description: The text to be scrubbed and truncated.
        max_length: The maximum allowed length of the output string. Use None for no limit.
        split_location: When truncating, this indicates how much of `max_length`
          should be taken from the beginning, with the remainder from the end
          and an ellipsis in between. Default is 1.0 (100% from beginning).
    """
    if not description:
        return ""
    if max_length is None:
        max_length = 1 << 30 # effectively no limit
    description = strip_html(description)
    # replace timestamps with 'XX:XX' to avoid hallucinations
    description = re.sub(r'(?:\d+:)?\d{1,2}:\d{2}', 'XX:XX', description)
    # shorten urls (keep scheme and domain)
    description = re.sub(r'(https?://[^/\s]+)/\S+', r'\1/...', description)
    # remove trailing whitespace and empty lines
    description = re.sub(r'(?:\s*\n)+', '\n', description)
    if len(description) > max_length:
        # truncate to max_length at the specified split location
        if max_length <= 3:
            return "..."
        # calc split points, taking into account the ellipsis length (3)
        split_location = max(0.0, min(1.0, split_location))
        clip_begin = round((max_length - 3) * split_location)
        clip_end = len(description) - (max_length - 3 - clip_begin)
        # adjust to word boundaries...
        if description[clip_begin] != ' ':
            word_begin = description.rfind(' ', 0, clip_begin)
            if word_begin >= 0:
                clip_begin = word_begin + 1 # include space
        if description[clip_end - 1] != ' ':
            clip_end = max(clip_end, description.find(' ', clip_end))
        description = description[:clip_begin] + "..." + description[clip_end:]
    return description

def indent_text(text: str, indent: str|int,
                indent_empty: bool = False,
                skip_first: bool = False) -> str:
    """Indent each line of the given text with the specified indent string.
    When indent_empty is False, empty lines, including trailing newline, are NOT
    indented.
    When skip_first is True, the first line of the text is not indented.
    """
    if not indent:
        return text
    if isinstance(indent, int):
        indent = ' ' * indent
    if not text:
        return "" if not indent_empty or skip_first else indent
    pattern = r'(^|\n)' if not skip_first else r'(\n)'
    if not indent_empty:
        pattern += r'(?!\n|$)'
    return re.sub(pattern, rf'\1{indent}', text)

def _clean_line(line: str, replace_with: str = ' ') -> str:
    """Remove control characters and trim the line."""
    return re.sub(r'[\x00-\x1f]+', replace_with, line).strip() if line else ""

def format_prompt_heading(heading: str, level: int, content: str = "", indent: int = 0) -> str:
    """Format a text heading with the specified level using markdown headings.
    Heading is trimmed.
    Returned string ends with a new line.
    """
    heading = _clean_line(heading) if heading else ""
    if not heading:
        raise ValueError("Heading cannot be empty")
    if level < 1:
        raise ValueError("Heading level must be at least 1")
    content = content.strip() if content else ""
    if content:
        content = content + "\n"
    return indent_text(f"{'#' * min(level, 6)} {heading}\n\n{content}", indent)

def format_prompt_section(title: str, content: str = "", indent: int = 0) -> str:
    """Format a text section using bold title using markdown section style.
    Both title and text are trimmed.
    Returned string ends with a new line.
    """
    title = _clean_line(title) if title else ""
    if not title:
        raise ValueError("Section title cannot be empty")
    content = content.strip() if content else ""
    if content:
        content = content + "\n"
    return indent_text(f"**{title}:**\n{content}", indent)

def format_prompt_list(lines: list[str], bullet: str = '-',
                       skip_empty: bool = True, indent: int = 0) -> str:
    """Format a list of lines as a markdown bullet list.
    Lines are trimmed. Empty lines are ignored when skip_empty is True.
    Returned string ends with a new line.
    """
    line_indent = ' ' * (len(bullet) + 1)
    bulleted_lines = [
        f"{bullet} {indent_text(trimmed, line_indent, skip_first=True)}"
        for line in lines if (trimmed := line.strip()) or not skip_empty
    ]
    return indent_text("\n".join(bulleted_lines) + "\n", indent)

def format_prompt_list_numbered(lines: list[str], start_at: int = 1,
                                number_format: str = "{0}.",
                                skip_empty: bool = True, indent: int = 0) -> str:
    """Format a list of lines as a markdown numbered list.
    Lines are trimmed. Empty lines are ignored when skip_empty is True.
    Returned string ends with a new line.
    """
    numbered_lines = []
    line_indent = ''
    for line in lines:
        line = line.strip()
        if skip_empty and not line:
            continue
        number = number_format.format(start_at + len(numbered_lines))
        if len(line_indent) != len(number) + 1:
            line_indent = ' ' * (len(number) + 1)
        numbered_lines.append(f"{number} {indent_text(line, line_indent, skip_first=True)}")
    return indent_text("\n".join(numbered_lines) + "\n", indent)

def format_prompt_code_block(code: str, language: str = "text", fence_char: str = "`", indent: int = 0) -> str:
    """Format a block of text as a markdown code block. Supports nested code blocks.
    Language is trimmed, code is not.
    Returned string ends with a new line.
    """
    if len(fence_char) != 1:
        raise ValueError("fence_char must be a single character")
    language = _clean_line(language, replace_with='-') if language else ""
    # find the longest sequence of fence_char at the beginning of a line (up to
    # 3 leading spaces) then wrap the code using a fence 1 longer than that.
    max_fence_len = 2
    for fence in re.findall(rf'^\s{{,3}}({re.escape(fence_char)}{{3,}})', code, flags=re.MULTILINE):
        max_fence_len = max(max_fence_len, len(fence))
    fence = fence_char * (max_fence_len + 1)
    return indent_text(f"{fence}{language}\n{code}\n{fence}\n", indent)

def format_detection_notes_block(notes: str|None) -> str:
    """Prompt block for per-feed operator notes, or "" when there are none.
    Returned string ends with 2 new lines.
    """
    # Operator text is trusted, so strip consecutive newlines only.
    notes = re.sub(r'(?:\s*\n)+', '\n', notes or '').strip()[:DETECTION_NOTES_MAX_CHARS]
    if not notes:
        return ""
    return (
        f"Podcast User Notes:\n{notes}\n\n"
    )

def strip_comments_from_prompt(prompt: str|None) -> str:
    """Remove HTML-style comments from the prompt following markdown
    conventions.

    Multiline comments can start at the beginning of a line (up to three
    leading spaces allowed), and single-line comments can appear anywhere.
    If a ML comment ends with a line break, the line break is also removed.

    HTML-style comments cannot be nested (the nested comment's end will
    terminate the outer comment).

    Examples:
        Text <!-- This is a single-line comment -->
        <!--
        This is a multiline comment
        -->
           <!-- This is a [multiline] comment (<=3 spaces after BOL) -->
            <!-- Not a comment (>3 spaces after BOL) -->
        Text <!--
            Not a comment (text before multiline comment)
            -->
    """
    if not prompt:
        return ""
    pattern = (r'^([ ]{0,3})<!--(?:.|\n)*?-->(?:\r?\n)?' # multiline (1 or more) comment
               r'|^([ ]{4,}<!--.*?-->)' # keep literal single-line comment
               r'|<!--.*?-->') # single-line comment (dot does NOT match newlines)
    return re.sub(pattern, r'\1\2', prompt, flags=re.MULTILINE)


def finalize_prompt(prompt: str) -> str:
    """Finalize the prompt string by removing unnecessary whitespace and
    redundant newlines.
    """
    prompt = prompt.strip() if prompt else ""
    if not prompt:
        return ""

    # remove trailing whitespace from all lines
    prompt = re.sub(r"[ \t]+\r?\n", "\n", prompt)
    # Collapse 3 or more consecutive newlines into two
    prompt = re.sub(r"(?:\r?\n){3,}", "\n\n", prompt)

    return prompt
