import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication, QSplashScreen
from PySide6.QtGui import QPixmap, QColor, QFontDatabase
from PySide6.QtCore import Qt

def _load_bundled_fonts():
    """Register every .ttf/.otf in ui/fonts/ with Qt's font database.
    Resilient to the folder being absent or empty — fonts fall back to
    system equivalents gracefully until bundled faces are present."""
    font_dir = Path(__file__).resolve().parent / "ui" / "fonts"
    if not font_dir.is_dir():
        return
    for f in font_dir.iterdir():
        if f.suffix.lower() in (".ttf", ".otf"):
            QFontDatabase.addApplicationFont(str(f))

def main():
    app = QApplication(sys.argv)

    # Required for QStandardPaths to resolve the correct per-user config dir
    # on all platforms (Windows: %LOCALAPPDATA%\SOFA Buddy,
    # Linux: ~/.config/SOFA Buddy, macOS: ~/Library/Preferences/SOFA Buddy)
    app.setOrganizationName("SOFA Buddy")
    app.setApplicationName("SOFA Buddy")

    _load_bundled_fonts()

    # Import theme after QApplication is created (avoids any Qt init ordering issues)
    from ui import theme as T

    # Set global application styles
    font = app.font()
    font.setFamily("Roboto") # Modern sans-serif; resolved from ui/fonts/ when bundled
    font.setPointSize(11)
    app.setFont(font)

    app.setStyleSheet(f"""
        QToolTip {{
            background-color: {T.BG_WIDGET};
            color: {T.FG};
            border: 1px solid {T.BG_ACTIVE};
            padding: 10px;
            max-width: 450px;
            font-size: 14px;
            font-family: "Roboto";
        }}
    """)

    # 1. Create and show Splash Screen immediately
    # Using a simple colored pixmap as a placeholder for a logo
    pixmap = QPixmap(400, 300)
    pixmap.fill(QColor(T.SPLASH_BG))

    splash = QSplashScreen(pixmap, Qt.WindowType.WindowStaysOnTopHint)
    splash.show()

    # Show initial message
    splash.showMessage(
        "Loading SOFA Buddy...",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignCenter,
        Qt.GlobalColor.white
    )
    app.processEvents() # Ensure splash is painted

    # 2. Defer import of Main Window until after splash is shown
    from ui.main_window import SofaBuddyMainWindow

    window = SofaBuddyMainWindow()

    # 3. Close splash and show main window
    splash.finish(window)
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
