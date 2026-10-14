"""Light, card-based look (reference: Toss Securities): white cards on a soft gray page, one blue
action color, large numbers, and Korean market colors (rise = red, fall = blue)."""

from __future__ import annotations

BG = "#F2F4F6"
CARD = "#FFFFFF"
TEXT = "#191F28"
TEXT2 = "#4E5968"
MUTED = "#8B95A1"
LINE = "#E5E8EB"
BLUE = "#3182F6"
BLUE_DARK = "#1B64DA"
BLUE_SOFT = "#E8F3FF"
RISE = "#F04452"
FALL = "#3182F6"
VERDICT = {"PASS": ("#03B26C", "#E5F8EF"), "HOLD": ("#FF8A00", "#FFF3E0"), "FAIL": ("#F04452", "#FEECEE"),
           "INVALID": ("#6B7684", "#F2F4F6")}

FONT = '"Pretendard", "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans CJK KR", sans-serif'

QSS = f"""
* {{ font-family: {FONT}; font-size: 14px; color: {TEXT}; }}
QMainWindow, QWidget#page, QScrollArea, QScrollArea > QWidget > QWidget#page {{ background: {BG}; }}
QWidget#sidebar {{ background: {CARD}; border-right: 1px solid {LINE}; }}
QWidget#sidebar QPushButton {{ text-align: left; padding: 11px 16px; border: none; border-radius: 10px;
    font-size: 15px; color: {TEXT2}; background: transparent; }}
QWidget#sidebar QPushButton:hover {{ background: {BG}; }}
QWidget#sidebar QPushButton:checked {{ background: {BG}; color: {TEXT}; font-weight: 700; }}
QLabel#logo {{ font-size: 20px; font-weight: 800; color: {TEXT}; padding: 6px 16px 18px 16px; }}
QFrame#card {{ background: {CARD}; border-radius: 18px; }}
QLabel#h1 {{ font-size: 26px; font-weight: 800; }}
QLabel#h2 {{ font-size: 19px; font-weight: 700; }}
QLabel#h3 {{ font-size: 16px; font-weight: 700; }}
QLabel#muted {{ color: {MUTED}; font-size: 13px; }}
QLabel#body2 {{ color: {TEXT2}; }}
QLabel#big {{ font-size: 26px; font-weight: 800; }}
QLabel#chip {{ background: {BG}; color: {TEXT2}; border-radius: 8px; padding: 3px 8px; font-size: 12px; font-weight: 600; }}
QPushButton {{ background: {BG}; color: {TEXT}; border: none; border-radius: 10px; padding: 10px 16px; font-weight: 600; }}
QPushButton:hover {{ background: {LINE}; }}
QPushButton:disabled {{ color: {MUTED}; }}
QPushButton#primary {{ background: {BLUE}; color: white; font-size: 15px; }}
QPushButton#primary:hover {{ background: {BLUE_DARK}; }}
QPushButton#primary:disabled {{ background: #B0D0FB; color: white; }}
QPushButton#secondary {{ background: {BLUE_SOFT}; color: {BLUE_DARK}; }}
QPushButton#secondary:hover {{ background: #D6E9FF; }}
QPushButton#link {{ background: transparent; color: {TEXT2}; padding: 6px 4px; font-weight: 500; }}
QPushButton#link:hover {{ color: {TEXT}; }}
QPushButton#seg {{ background: transparent; color: {MUTED}; padding: 6px 12px; border-radius: 8px; }}
QPushButton#seg:checked {{ background: {BG}; color: {TEXT}; }}
QLineEdit, QComboBox, QSpinBox, QDateEdit {{ background: {BG}; border: 1px solid transparent; border-radius: 10px; padding: 9px 12px; }}
QLineEdit:focus {{ border: 1px solid {BLUE}; background: {CARD}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QProgressBar {{ background: {LINE}; border: none; border-radius: 3px; max-height: 6px; text-align: center; }}
QProgressBar::chunk {{ background: {BLUE}; border-radius: 3px; }}
QTableWidget {{ background: {CARD}; border: none; border-radius: 12px; gridline-color: {LINE};
    alternate-background-color: #FAFBFC; selection-background-color: {BLUE_SOFT}; selection-color: {TEXT}; }}
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
QScrollBar::handle:vertical {{ background: #D1D6DB; border-radius: 5px; min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QMessageBox {{ background: {CARD}; }}
"""


def signed_color(x: float | None) -> str:
    if x is None or x != x or abs(x) < 1e-12:
        return TEXT
    return RISE if x > 0 else FALL


def apply(app) -> None:
    """Light palette regardless of the Windows theme, plus the stylesheet."""
    from PySide6.QtGui import QColor, QPalette

    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in ((QPalette.Window, BG), (QPalette.Base, CARD), (QPalette.AlternateBase, "#FAFBFC"),
                        (QPalette.Text, TEXT), (QPalette.WindowText, TEXT), (QPalette.Button, BG),
                        (QPalette.ButtonText, TEXT), (QPalette.Highlight, BLUE), (QPalette.HighlightedText, "#FFFFFF"),
                        (QPalette.ToolTipBase, CARD), (QPalette.ToolTipText, TEXT), (QPalette.PlaceholderText, MUTED)):
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
