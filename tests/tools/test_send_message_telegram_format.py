"""Regression tests for the standalone Telegram sender's HTML-vs-MarkdownV2 detection.

The defect class (operator-visible "escaping" in BAF receipts): a loose angle-bracket probe
(``<[a-zA-Z/][^>]*>``) routed literal text — path tokens like ``runs/<stamp>/``, placeholders
like ``<tag>`` — into unescaped HTML mode. Telegram then rejected the whole message and the
send silently fell back to plain text, stripping every markdown formatting character into the
visible text. Detection now requires a supported Telegram HTML tag.

Contracts:
- literal tag-shaped text   -> MarkdownV2 (adapter formatting), bytes delivered verbatim;
- supported HTML tags       -> unchanged HTML passthrough (previous behavior);
- the character matrix from the investigation becomes the regression set.
"""

import pytest
from telegram.constants import ParseMode

from tools.send_message_senders import _telegram_format

# Literal text that only LOOKS like markup; must take the MarkdownV2 path.
LITERAL_ANGLE_TEXT = [
    "<stamp>",
    "</x>",
    "<a b>",
    "<sid>",
    "runs/<stamp>/",
    "artifact runs/<stamp>/receipt.json sealed",
    "<notatag>",
    "<5=30m.",
    "a < b > c",
    "<<<<",
    "<>",
    "<...>",
    "<'a'a'a'a>",
    "trailing <",
    "<T>U",
]

# Actual Telegram HTML subset; must keep the HTML passthrough.
SUPPORTED_HTML = [
    "<b>bold</b>",
    "<strong>bold</strong>",
    "<i>it</i>",
    "<em>it</em>",
    "<u>u</u>",
    "<ins>u</ins>",
    "<s>s</s>",
    "<strike>s</strike>",
    "<del>d</del>",
    "<B>upper</B>",
    "<tg-spoiler>s</tg-spoiler>",
    '<span class="tg-spoiler">s</span>',
    '<a href="https://example.com/a?b=1">ex</a>',
    "<code>c()</code>",
    '<pre><code class="language-python">c()</code></pre>',
    "<blockquote>q</blockquote>",
    "<blockquote expandable>q</blockquote>",
]


def _unescape_mdv2(text: str) -> str:
    """Drop MarkdownV2 backslash escapes to recover the text Telegram displays."""
    import re
    return re.sub(r"\\(.)", r"\1", text)


@pytest.mark.parametrize("message", LITERAL_ANGLE_TEXT)
def test_literal_angle_text_routes_to_markdownv2(message):
    formatted, parse_mode, has_html = _telegram_format(message)
    assert parse_mode == ParseMode.MARKDOWN_V2
    assert has_html is False
    # The angle-bracket text survives formatting (never HTML-comment/entity-mangled).
    assert "<" in formatted
    assert "&lt;" not in formatted


@pytest.mark.parametrize("message", SUPPORTED_HTML)
def test_supported_html_stays_html_passthrough(message):
    formatted, parse_mode, has_html = _telegram_format(message)
    assert parse_mode == ParseMode.HTML
    assert has_html is True
    assert formatted == message


def test_receipt_shaped_body_keeps_path_token_verbatim():
    """The operator-visible regression: a receipt whose quoted evidence line carries a path
    token must deliver through MarkdownV2 with the token intact after unescaping."""
    message = (
        "**\u25c6 ANSWER \u00b7 TS2 \u2192 TS8 \u00b7 12:09**\n"
        "`question \u00b7 baf-escaping \u00b7 TS2#001 \u00b7 \u25cb pending \u00b7 response ts8`\n"
        "> evidence sealed under ~/nouma-qa/runs/<stamp>/ before delivery"
    )
    formatted, parse_mode, has_html = _telegram_format(message)
    assert parse_mode == ParseMode.MARKDOWN_V2
    assert has_html is False
    assert "runs/<stamp>/" in _unescape_mdv2(formatted)


def test_mixed_supported_and_literal_keeps_html_path():
    """A message with a real tag keeps the old HTML contract; the literal run is untouched
    (fallback behavior for such mixed content is unchanged by this fix)."""
    message = "<b>bold</b> and literal <stamp> too"
    formatted, parse_mode, has_html = _telegram_format(message)
    assert parse_mode == ParseMode.HTML
    assert has_html is True
    assert formatted == message
