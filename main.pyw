import os

# Pin BLAS/OpenMP thread counts to 1 before anything imports numpy/scipy/tslearn.
# The parent process runs global averaging (Soft-DTW barycenter) in-process; its
# concurrency is bounded explicitly by the chunk-pool worker count, so BLAS must
# not additionally fan out threads and oversubscribe cores. setdefault respects
# any value the user already exported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication, QSplashScreen
from PySide6.QtGui import QPixmap, QFontDatabase
from PySide6.QtCore import Qt, QTimer

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
    splash_path = Path(__file__).resolve().parent / "ui" / "sb_splash.png"
    pixmap = QPixmap(str(splash_path)).scaled(
        400, 400,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )

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

    # One-time settings-migration notice: queued for when the event loop starts,
    # so it appears above the visible main window, not behind the splash.
    QTimer.singleShot(0, window.show_startup_notices)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
