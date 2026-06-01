import sys
from PyQt6.QtWidgets import QApplication, QSplashScreen
from PyQt6.QtGui import QPixmap, QColor
from PyQt6.QtCore import Qt

def main():
    app = QApplication(sys.argv)
    
    # Set global application styles
    font = app.font()
    font.setFamily("Roboto") # Modern sans-serif
    font.setPointSize(11)
    app.setFont(font)
    
    app.setStyleSheet("""
        QToolTip {
            background-color: #333333;
            color: white;
            border: 1px solid #666666;
            padding: 10px;
            max-width: 450px;
            font-size: 14px;
            font-family: "Roboto";
        }
    """)

    # 1. Create and show Splash Screen immediately
    # Using a simple colored pixmap as a placeholder for a logo
    pixmap = QPixmap(400, 300)
    pixmap.fill(QColor("#2b2b2b"))
    
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
