"""
Shared output-filename template rendering.

Output filenames across SOFA mastering and DFHRTF analysis are built from
user-editable templates containing `{var}` substitution tokens (the
variables are listed in README.md, "Filename Template Variables"). This module is the single place
where that substitution happens, so SOFA naming, DFHRTF text, DFHRTF graphs,
per-SOFA averages and global averages all agree on the semantics.

Semantics (deliberately narrow, matching the historical `.replace()` chains):

  * All tokens are substituted in a *single* left-to-right pass over the
    template. Substituted values are never re-scanned, so a value that happens
    to contain something like ``{eq}`` survives verbatim, and the result does
    not depend on the order of the `tokens` mapping.
  * A token appearing in the template but absent from `tokens` (and not in
    `drop`) is left in place verbatim. Templates are user input; silently
    deleting an unrecognized token would produce surprising filenames, and
    raising would turn a typo into a failed run.
  * `drop` removes a token together with one adjacent separator character
    (``_``, ``-`` or a space), which is what "this output has no algorithm
    name" means for a template like ``{filename}_{algo}``.

This module performs *no* path sanitization: token values containing path
separators or ``..`` flow straight through into the rendered name. See the
xfail test in tests/test_naming.py.
"""

import re

# Separator characters that may glue a dropped token to the rest of the name.
_SEPARATORS = "_\\- "

# One `{token}` occurrence. Token names never contain braces.
_TOKEN_RE = re.compile(r"\{([^{}]*)\}")


def _drop_token(text, name):
    """Remove every ``{name}`` occurrence plus one adjacent separator, if any.

    A preceding separator is preferred over a trailing one, so
    ``"{filename}_{algo}_x"`` renders as ``"{filename}_x"`` rather than
    ``"{filename}__x"``.
    """
    token = re.escape("{" + name + "}")
    pattern = re.compile(
        "[" + _SEPARATORS + "]" + token   # separator + token
        + "|" + token + "[" + _SEPARATORS + "]"  # token + separator
        + "|" + token                     # bare token
    )
    return pattern.sub("", text)


def render_filename(template, tokens, *, ext=None, drop=frozenset()):
    """Render a filename template.

    Args:
        template: template string, e.g. ``"{filename}_{samplerate}_{eq}"``.
        tokens: mapping of *unprefixed* token names to values, e.g.
            ``{"filename": "subject01", "samplerate": 48000}``. Values are
            coerced with ``str()``.
        ext: optional extension (including the leading dot) appended only when
            the rendered name does not already end with it, compared
            case-insensitively.
        drop: iterable of token names to remove from the template entirely,
            along with one adjacent ``_``/``-``/space separator. Dropping wins
            over substitution: a name in both `drop` and `tokens` is dropped.

    Returns:
        The rendered filename (no directory component).
    """
    result = template

    for name in drop:
        result = _drop_token(result, name)

    def _sub(match):
        name = match.group(1)
        if name in drop or name not in tokens:
            return match.group(0)  # leave unknown/dropped tokens verbatim
        return str(tokens[name])

    result = _TOKEN_RE.sub(_sub, result)

    if ext and not result.lower().endswith(ext.lower()):
        result += ext

    return result
