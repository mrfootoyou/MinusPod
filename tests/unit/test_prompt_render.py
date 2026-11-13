"""Tests for the prompt placeholder substitution refactor."""
import sqlite3
from unittest.mock import MagicMock

import pytest


from ad_detector import get_static_system_prompt
from utils.prompt import (
    SPONSOR_DATABASE_HEADER,
    format_prompt_code_block,
    format_prompt_heading,
    format_prompt_list,
    format_prompt_list_numbered,
    format_prompt_section,
    format_sponsor_block,
    indent_text,
    render_prompt,
)
from database import (
    DEFAULT_RESURRECT_PROMPT,
    DEFAULT_REVIEW_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_VERIFICATION_PROMPT,
)

class TestIndentText:
    def test_empty_string(self):
        assert indent_text("", 0) == ""
        assert indent_text("", "") == ""
        assert indent_text("", 2) == ""
        assert indent_text("", "  ") == ""

    def test_indent_text_with_integer_indent(self):
        assert indent_text("line1", 2) == "  line1"
        assert indent_text("line1\nline2", 2) == "  line1\n  line2"
        assert indent_text("line1\nline2\n", 2) == "  line1\n  line2\n"

    def test_indent_text_with_string_indent(self):
        assert indent_text("line1", "  ") == "  line1"
        assert indent_text("line1\nline2", "  ") == "  line1\n  line2"
        assert indent_text("line1\nline2\n", "  ") == "  line1\n  line2\n"

    def test_indent_empty_lines(self):
        assert indent_text("", "  ", indent_empty=True) == "  "
        assert indent_text("\n\n", "  ", indent_empty=True) == "  \n  \n  "

    def test_skip_first_line(self):
        assert indent_text("Line1\nLine2\n", "  ", skip_first=True) == "Line1\n  Line2\n"
        assert indent_text("", "  ", skip_first=True, indent_empty=True) == ""
        assert indent_text("\n\n", "  ", skip_first=True, indent_empty=True) == "\n  \n  "

class TestFormatPromptHeading:
    def test_basics(self):
        assert format_prompt_heading("Test", 1) == "# Test\n\n"
        assert format_prompt_heading("\n  Test \n", 1) == "# Test\n\n"
        assert format_prompt_heading("Test", 2) == "## Test\n\n"
        assert format_prompt_heading("Test", 6) == "###### Test\n\n"
        assert format_prompt_heading("Test", 7) == "###### Test\n\n"

    def test_clean_line(self):
        assert format_prompt_heading("Te\t\tst", 1) == "# Te st\n\n"

    def test_with_indent(self):
        assert format_prompt_heading("Test", 1, indent=2) == "  # Test\n\n"

    def test_empty_heading_raises(self):
        with pytest.raises(ValueError):
            format_prompt_heading("", 1)
        with pytest.raises(ValueError):
            format_prompt_heading(" \t", 1)

    def test_heading_level_too_low_raises(self):
        with pytest.raises(ValueError):
            format_prompt_heading("Test", 0)

class TestFormatPromptSection:
    def test_basics(self):
        assert format_prompt_section("Section") == "**Section:**\n"
        assert format_prompt_section("\n  Section \n") == "**Section:**\n"
        assert format_prompt_section("Section", "text") == "**Section:**\ntext\n"
        assert format_prompt_section("Section", "\n  text \n ") == "**Section:**\ntext\n"

    def test_clean_line(self):
        assert format_prompt_section("Sec\t\ttion", "\n  text \n ") == "**Sec tion:**\ntext\n"

    def test_with_indent(self):
        assert format_prompt_section("Section", "text", indent=2) == "  **Section:**\n  text\n"
        assert format_prompt_section("Section", "line1\nline2", indent=2) == "  **Section:**\n  line1\n  line2\n"

    def test_empty_section_title_raises(self):
        with pytest.raises(ValueError):
            format_prompt_section("", "text")
        with pytest.raises(ValueError):
            format_prompt_section(" \t", "text")

class TestFormatPromptList:
    def test_basics(self):
        assert format_prompt_list([]) == "\n"
        assert format_prompt_list(["item1"]) == "- item1\n"
        assert format_prompt_list(["item1", "item2"]) == "- item1\n- item2\n"
        assert format_prompt_list(["\n item1 \n ", "  item2\nline2  \n"]) == "- item1\n- item2\n  line2\n"

    def test_with_custom_bullet(self):
        assert format_prompt_list(["item1", "item2"], bullet='*') == "* item1\n* item2\n"

    def test_with_not_skip_empty_items(self):
        assert format_prompt_list(["item1", "", "item3"], skip_empty=False) == "- item1\n- \n- item3\n"

    def test_with_indent(self):
        assert format_prompt_list([], indent=2) == "\n"
        assert format_prompt_list(["item1"], indent=2) == "  - item1\n"
        assert format_prompt_list(["item1", "item2\n\nline3"], indent=2) == "  - item1\n  - item2\n\n    line3\n"

class TestFormatPromptListNumbered:
    def test_basics(self):
        assert format_prompt_list_numbered([]) == "\n"
        assert format_prompt_list_numbered(["item1"]) == "1. item1\n"
        assert format_prompt_list_numbered(["item1", "item2"]) == "1. item1\n2. item2\n"
        assert format_prompt_list_numbered(["\n item1 \n ", "  item2\nline2  \n"]) == "1. item1\n2. item2\n   line2\n"

    def test_with_start_at(self):
        assert format_prompt_list_numbered(["item1", "item2"], start_at=3) == "3. item1\n4. item2\n"
        assert format_prompt_list_numbered(["item1", "item2"], start_at=-20) == "-20. item1\n-19. item2\n"

    def test_with_custom_number_format(self):
        assert format_prompt_list_numbered(["item1", "item2"], number_format='{0})') == "1) item1\n2) item2\n"

    def test_with_not_skip_empty_items(self):
        assert format_prompt_list_numbered(["item1", "", "item3"], skip_empty=False) == "1. item1\n2. \n3. item3\n"

    def test_with_indent(self):
        assert format_prompt_list_numbered([], indent=2) == "\n"
        assert format_prompt_list_numbered(["item1"], indent=2) == "  1. item1\n"
        assert format_prompt_list_numbered(["item1", "item2\n\nline3"], start_at=100, indent=2) == "  100. item1\n  101. item2\n\n       line3\n"

class TestFormatPromptCodeBlock:
    def test_basics(self):
        assert format_prompt_code_block('') == '```text\n\n```\n'
        assert format_prompt_code_block('print("Hello")') == '```text\nprint("Hello")\n```\n'

    def test_preserves_whitespace(self):
        assert format_prompt_code_block('  print("Hello")\n  exit 1') == \
            '```text\n  print("Hello")\n  exit 1\n```\n'

    def test_with_language(self):
        assert format_prompt_code_block('foo', language='') == '```\nfoo\n```\n'
        assert format_prompt_code_block('', language='sh') == '```sh\n\n```\n'
        assert format_prompt_code_block('print("Hello")', language='sh') == \
            '```sh\nprint("Hello")\n```\n'
        assert format_prompt_code_block('', language='foo\t\nbar') == '```foo-bar\n\n```\n'

    def test_with_indent(self):
        assert format_prompt_code_block('print("Hello")', indent=2) == \
            "  ```text\n  print(\"Hello\")\n  ```\n"

    def test_with_fence_char(self):
        assert format_prompt_code_block('print("Hello")', fence_char='~') == \
            "~~~text\nprint(\"Hello\")\n~~~\n"

    def test_with_nested_code_block(self):
        markdown = (
            'Example:\n'
            '~~~sh\n'
            'print("Hello")\n'
            '~~~'
        )
        text = format_prompt_code_block(markdown, language='markdown', 
                                       fence_char='~', indent=2)
        assert text == (
            '  ~~~~markdown\n'
            '  Example:\n'
            '  ~~~sh\n'
            '  print("Hello")\n'
            '  ~~~\n'
            '  ~~~~\n'
        )

    def test_invalid_fence_char_raises(self):
        with pytest.raises(ValueError):
            format_prompt_code_block("code", fence_char='')
        with pytest.raises(ValueError):
            format_prompt_code_block("code", fence_char='aa')

        
def test_render_prompt_substitutes_single_brace_placeholders():
    out = render_prompt('Hello {name}', name='world')
    assert out == 'Hello world'


def test_render_prompt_preserves_literal_double_braces():
    """Double-braced JSON examples in default prompts must not be touched."""
    out = render_prompt('JSON: {"start": 1.0} sub={x}', x='Y')
    assert out == 'JSON: {"start": 1.0} sub=Y'


def test_render_prompt_drops_missing_placeholders_silently():
    """Variables without a corresponding placeholder are no-ops (the user
    intentionally edited the placeholder out)."""
    out = render_prompt('Hello {a}', a='X', b='IGNORED')
    assert out == 'Hello X'


def test_render_prompt_absent_placeholder_means_no_insertion():
    """Key behavior change vs legacy append: no placeholder = no content."""
    out = render_prompt('Just plain text', sponsor_database='SHOULD NOT APPEAR')
    assert out == 'Just plain text'


def test_format_sponsor_block_empty_returns_empty():
    assert format_sponsor_block('') == ''


def test_format_sponsor_block_wraps_with_header():
    block = format_sponsor_block('AG1, BetterHelp')
    assert block.startswith(SPONSOR_DATABASE_HEADER)
    assert 'AG1, BetterHelp' in block


def test_default_system_prompt_has_sponsor_placeholder():
    assert '{sponsor_database}' in DEFAULT_SYSTEM_PROMPT
    assert '{sponsor_watchlists}' in DEFAULT_SYSTEM_PROMPT
    assert '{output_format}' in DEFAULT_SYSTEM_PROMPT
    assert '{examples}' in DEFAULT_SYSTEM_PROMPT


def test_default_verification_prompt_has_sponsor_placeholder():
    assert '{sponsor_database}' in DEFAULT_VERIFICATION_PROMPT
    assert '{sponsor_watchlists}' in DEFAULT_VERIFICATION_PROMPT
    assert '{output_format}' in DEFAULT_VERIFICATION_PROMPT
    assert '{examples}' in DEFAULT_VERIFICATION_PROMPT


def test_default_review_prompt_has_both_placeholders():
    assert '{sponsor_database}' in DEFAULT_REVIEW_PROMPT
    assert '{max_boundary_shift_seconds}' in DEFAULT_REVIEW_PROMPT


def test_default_resurrect_prompt_has_sponsor_placeholder():
    assert '{sponsor_database}' in DEFAULT_RESURRECT_PROMPT


def test_get_static_system_prompt_substitutes_seed_sponsors():
    out = get_static_system_prompt()
    assert '{sponsor_database}' not in out  # placeholder consumed
    assert 'DYNAMIC SPONSOR DATABASE' in out  # block was inserted
    # Seed sponsors include a few well-known ones
    assert 'BetterHelp' in out or 'Athletic Greens' in out


def test_render_with_sponsors_empty_list_drops_block():
    """When the dynamic sponsor list is empty, no header should appear."""
    from ad_detector import AdDetector
    detector = AdDetector.__new__(AdDetector)
    # Bypass the sponsor_service property by stubbing the helper directly
    detector._get_sponsor_list_safely = lambda: ''
    detector.db = MagicMock()
    detector.db.get_setting.return_value = 'true'
    out = detector._render_with_sponsors('Body{sponsor_database}END', 'seed_sponsors_detection')
    assert out == 'BodyEND'


def test_render_with_sponsors_non_empty_list_inserts_block():
    from ad_detector import AdDetector
    detector = AdDetector.__new__(AdDetector)
    detector._get_sponsor_list_safely = lambda: 'AG1, Squarespace'
    detector.db = MagicMock()
    detector.db.get_setting.return_value = 'true'
    out = detector._render_with_sponsors('Body{sponsor_database}END', 'seed_sponsors_detection')
    assert 'DYNAMIC SPONSOR DATABASE' in out
    assert 'AG1, Squarespace' in out
    assert 'Body' in out and 'END' in out


# ---------- Migration tests ----------

def test_migration_appends_placeholder_to_customized_prompts(tmp_path):
    """Existing user-customized prompts get {sponsor_database} appended."""
    from database import Database
    Database._instance = None

    # Pre-create a DB with customized prompts that lack the placeholder
    db_path = tmp_path / "podcast.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE settings (
            key TEXT PRIMARY KEY,
            value TEXT,
            is_default INTEGER DEFAULT 0,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE podcasts (id INTEGER PRIMARY KEY, slug TEXT);
        INSERT INTO podcasts(slug) VALUES ('seed-pod');
        INSERT INTO settings (key, value, is_default)
            VALUES ('system_prompt', 'Custom system without placeholder', 0);
        INSERT INTO settings (key, value, is_default)
            VALUES ('verification_prompt', 'Custom verification without placeholder', 0);
    """)
    conn.commit()
    conn.close()

    db = Database(data_dir=str(tmp_path))

    sp = db.get_setting('system_prompt')
    vp = db.get_setting('verification_prompt')
    assert sp.endswith('{sponsor_database}')
    assert vp.endswith('{sponsor_database}')
    assert db.get_setting('_review_prompt_migrated') == 'true'


def test_migration_is_idempotent(tmp_path):
    """Running the migration twice does not double-append the placeholder."""
    from database import Database
    Database._instance = None

    db_path = tmp_path / "podcast.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE settings (
            key TEXT PRIMARY KEY,
            value TEXT,
            is_default INTEGER DEFAULT 0,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE podcasts (id INTEGER PRIMARY KEY, slug TEXT);
        INSERT INTO podcasts(slug) VALUES ('seed-pod');
        INSERT INTO settings (key, value, is_default)
            VALUES ('system_prompt', 'Custom no placeholder', 0);
    """)
    conn.commit()
    conn.close()

    db = Database(data_dir=str(tmp_path))
    first = db.get_setting('system_prompt')

    Database._instance = None
    db = Database(data_dir=str(tmp_path))
    second = db.get_setting('system_prompt')

    assert first == second
    # Exactly one placeholder, not two
    assert first.count('{sponsor_database}') == 1


def test_default_flagged_prompts_are_owned_by_the_refresh(tmp_path):
    """is_default=1 rows belong to the refresh path, which re-points them at
    the text this version ships. The placeholder backfill never has to touch
    them, since the shipped prompt already carries the placeholder.
    """
    from database import Database
    Database._instance = None

    fixture_value = (
        'Custom default-flagged prompt with TAGLINE and PLATFORM-INSERTED ADS '
        'and brand tagline ads and LABELLED AUDIO CUES keywords so prior '
        'migrations skip it.'
    )
    db_path = tmp_path / "podcast.db"
    conn = sqlite3.connect(str(db_path))
    # fixture_value is a hardcoded literal above, not external input.
    script = f"""
        CREATE TABLE settings (
            key TEXT PRIMARY KEY,
            value TEXT,
            is_default INTEGER DEFAULT 0,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE podcasts (id INTEGER PRIMARY KEY, slug TEXT);
        INSERT INTO podcasts(slug) VALUES ('seed-pod');
        INSERT INTO settings (key, value, is_default)
            VALUES ('system_prompt', '{fixture_value}', 1);
    """  # noqa: S608
    conn.executescript(script)
    conn.commit()
    conn.close()

    db = Database(data_dir=str(tmp_path))
    sp = db.get_setting('system_prompt')
    # An untouched row tracks the shipped default rather than keeping whatever
    # prompt happened to be in the database when it was created.
    from utils.constants import DEFAULT_SYSTEM_PROMPT
    assert sp == DEFAULT_SYSTEM_PROMPT
    assert sp != fixture_value
    # The shipped prompt carries the placeholder, rendered at call time.
    assert '{sponsor_database}' in sp


def test_migration_skips_already_placeholdered_prompts(tmp_path):
    """Customized prompts that already include the placeholder stay verbatim."""
    from database import Database
    Database._instance = None

    db_path = tmp_path / "podcast.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE settings (
            key TEXT PRIMARY KEY,
            value TEXT,
            is_default INTEGER DEFAULT 0,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE podcasts (id INTEGER PRIMARY KEY, slug TEXT);
        INSERT INTO podcasts(slug) VALUES ('seed-pod');
        INSERT INTO settings (key, value, is_default)
            VALUES ('system_prompt', 'My prompt {sponsor_database} done', 0);
    """)
    conn.commit()
    conn.close()

    db = Database(data_dir=str(tmp_path))
    sp = db.get_setting('system_prompt')
    assert sp == 'My prompt {sponsor_database} done'
