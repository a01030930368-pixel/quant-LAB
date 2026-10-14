"""Dark, card-based look (reference: Toss Securities dark mode): raised dark cards on a near-black
page, one blue action color, large numbers, and Korean market colors (rise = red, fall = blue)."""

from __future__ import annotations

BG = "#101013"
CARD = "#1C1C22"
RAISED = "#2A2A33"       # inputs, chips and secondary buttons on a card
TEXT = "#F2F2F5"
TEXT2 = "#B0B0BA"
MUTED = "#7B7B87"
LINE = "#2C2C35"
BLUE = "#3182F6"
BLUE_DARK = "#4B91F7"    # hover: a touch lighter on dark backgrounds
BLUE_SOFT = "#1B2B45"
RISE = "#F04452"
FALL = "#4B91F7"
VERDICT = {"PASS": ("#2FCA87", "#14322A"), "HOLD": ("#FFA53D", "#3A2A14"), "FAIL": ("#FF5C69", "#3D1A1F"),
           "INVALID": ("#9E9EA8", "#2A2A33")}

FONT = '"Pretendard", "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans CJK KR", sans-serif'

QSS = f"""
* {{ font-family: {FONT}; font-size: 14px; color: {TEXT}; }}
QMainWindow, QWidget#page, QScrollArea, QScrollArea > QWidget > QWidget#page {{ background: {BG}; }}
QWidget#sidebar {{ background: {CARD}; border-right: 1px solid {LINE}; }}
QWidget#sidebar QPushButton {{ text-align: left; padding: 11px 16px; border: none; border-radius: 10px;
    font-size: 15px; color: {TEXT2}; background: transparent; }}
QWidget#sidebar QPushButton:hover {{ background: {RAISED}; }}
QWidget#sidebar QPushButton:checked {{ background: {RAISED}; color: {TEXT}; font-weight: 700; }}
QLabel#logo {{ font-size: 20px; font-weight: 800; color: {TEXT}; padding: 6px 16px 18px 16px; }}
QFrame#card {{ background: {CARD}; border-radius: 18px; }}
QLabel#h1 {{ font-size: 26px; font-weight: 800; }}
QLabel#h2 {{ font-size: 19px; font-weight: 700; }}
QLabel#h3 {{ font-size: 16px; font-weight: 700; }}
QLabel#muted {{ color: {MUTED}; font-size: 13px; }}
QLabel#body2 {{ color: {TEXT2}; }}
QLabel#big {{ font-size: 26px; font-weight: 800; }}
QLabel#chip {{ background: {RAISED}; color: {TEXT2}; border-radius: 8px; padding: 3px 8px; font-size: 12px; font-weight: 600; }}
QPushButton {{ background: {RAISED}; color: {TEXT}; border: none; border-radius: 10px; padding: 10px 16px; font-weight: 600; }}
QPushButton:hover {{ background: #34343F; }}
QPushButton:disabled {{ color: {MUTED}; }}
QPushButton#primary {{ background: {BLUE}; color: white; font-size: 15px; }}
QPushButton#primary:hover {{ background: {BLUE_DARK}; }}
QPushButton#primary:disabled {{ background: #22406B; color: {MUTED}; }}
QPushButton#secondary {{ background: {BLUE_SOFT}; color: #7FB2FF; }}
QPushButton#secondary:hover {{ background: #223A5E; }}
QPushButton#link {{ background: transparent; color: {TEXT2}; padding: 6px 4px; font-weight: 500; }}
QPushButton#link:hover {{ color: {TEXT}; }}
QPushButton#seg {{ background: transparent; color: {MUTED}; padding: 6px 12px; border-radius: 8px; }}
QPushButton#seg:checked {{ background: {RAISED}; color: {TEXT}; }}
QLineEdit, QComboBox, QSpinBox, QDateEdit {{ background: {RAISED}; color: {TEXT}; border: 1px solid transparent; border-radius: 10px; padding: 9px 12px; }}
QLineEdit:focus {{ border: 1px solid {BLUE}; background: {CARD}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QProgressBar {{ background: {LINE}; border: none; border-radius: 3px; max-height: 6px; text-align: center; }}
QProgressBar::chunk {{ background: {BLUE}; border-radius: 3px; }}
QTableWidget {{ background: {CARD}; border: none; border-radius: 12px; gridline-color: {LINE};
    alternate-background-color: #212128; selection-background-color: {BLUE_SOFT}; selection-color: {TEXT}; }}
QHeaderView::section {{ background: {CARD}; color: {MUTED}; border: none; border-bottom: 1px solid {LINE};
    padding: 8px 6px; font-weight: 600; font-size: 12px; }}
QTabWidget::pane {{ border: none; background: {BG}; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 8px 14px; font-weight: 600; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {TEXT}; }}
QGroupBox {{ background: {CARD}; border: none; border-radius: 14px; margin-top: 18px; padding: 14px; font-weight: 700; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 6px; color: {TEXT2}; }}
QPlainTextEdit {{ background: {CARD}; border: none; border-radius: 12px; }}
QStatusBar {{ background: {CARD}; color: {TEXT2}; border-top: 1px solid {LINE}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; }}
QScrollBar::handle:vertical {{ background: #3A3A44; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QMessageBox {{ background: {CARD}; }}
QComboBox QAbstractItemView {{ background: {CARD}; color: {TEXT}; selection-background-color: {BLUE_SOFT}; border: 1px solid {LINE}; }}
QToolTip {{ background: {RAISED}; color: {TEXT}; border: 1px solid {LINE}; }}
QTableWidget QTableCornerButton::section {{ background: {CARD}; border: none; }}
"""


def signed_color(x: float | None) -> str:
    if x is None or x != x or abs(x) < 1e-12:
        return TEXT
    return RISE if x > 0 else FALL


def apply(app) -> None:
    """Dark palette regardless of the Windows theme, plus the stylesheet and dark chart colors."""
    from PySide6.QtGui import QColor, QPalette

    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in ((QPalette.Window, BG), (QPalette.Base, CARD), (QPalette.AlternateBase, "#212128"),
                        (QPalette.Text, TEXT), (QPalette.WindowText, TEXT), (QPalette.Button, RAISED),
                        (QPalette.ButtonText, TEXT), (QPalette.Highlight, BLUE), (QPalette.HighlightedText, "#FFFFFF"),
                        (QPalette.ToolTipBase, RAISED), (QPalette.ToolTipText, TEXT), (QPalette.PlaceholderText, MUTED)):
        pal.setColor(role, QColor(color))
    app.setPalette(pal)
    app.setStyleSheet(QSS)
    import logging

    import matplotlib

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # missing fallbacks are expected
    # Korean labels in charts: first installed font wins, glyph by glyph
    matplotlib.rcParams["font.family"] = ["Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans CJK KR", "Noto Sans CJK JP",
                                            "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

