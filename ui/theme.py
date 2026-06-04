# ui/theme.py — single source of truth for all UI colors and sizing constants.
# Consumers import this module as T: `from ui import theme as T`
# QSS f-strings must double literal braces: {{ and }}

# --- Backgrounds (dark → light) ---
BG_CONSOLE  = "#111111"   # console pane
BG_BASE     = "#222222"   # main window background
BG_WIDGET   = "#333333"   # list panes, tooltip background
BG_CONTROL  = "#444444"   # default control background
BG_ACTIVE   = "#666666"   # nav button :checked; line-edit underline
SPLASH_BG   = "#2b2b2b"   # splash screen placeholder pixmap

# --- Accent ---
ACCENT       = "#286C9E"  # selected toggle / primary action button
ACCENT_LIGHT = "#5aacdb"  # selected toggle border + dropdown arrow
ACCENT_FILL  = "#355775"  # slider value-bar fill

# --- Status ---
DANGER = "#D9534F"        # stop / abort action button

# --- Foreground ---
FG          = "#FFFFFF"   # normal text
FG_CONSOLE  = "#EEEEEE"   # slightly dimmed console output text
FG_DISABLED = "#111111"   # disabled control text

# --- Borders ---
BORDER_UNSEL = "#888888"  # unselected toggle left-border accent

# --- Disabled control backgrounds (two distinct cases) ---
DISABLED_UNSEL_BG = BG_CONTROL  # "#444444" — disabled unselected control
DISABLED_SEL_BG   = "#404952"   # disabled selected-by-default control

# --- Error text (console + append_error) ---
ERROR_TEXT = "#E06666"

# --- Sizing ---
CONTROL_HEIGHT = 40   # px; shared row height for all interactive controls
