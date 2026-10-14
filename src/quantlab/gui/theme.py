"""Trading-terminal look (design option B): near-black page, flat bordered panes, an amber
accent, numbers in a fixed-width font so columns line up, and Korean market colors (rise = red,
fall = blue)."""

from __future__ import annotations

BG = "#0A0C0F"
CARD = "#11151A"         # panes
RAISED = "#1A2027"       # inputs, chips and secondary buttons inside a pane
TEXT = "#D9DEE5"
TEXT2 = "#A7B0BC"
MUTED = "#7A8491"
LINE = "#1F262E"
ACCENT = "#F2A93B"       # amber: primary action, selected tab, strategy line
ACCENT_HOVER = "#F7BC5E"
ACCENT_SOFT = "#2A2216"
BLUE = ACCENT            # older name, kept for callers
BLUE_SOFT = ACCENT_SOFT
RISE = "#FF5A5F"
FALL = "#3B8CFF"
OK = "#2FBF71"
VERDICT = {"PASS": ("#2FBF71", "#11291D"), "HOLD": ("#F2A93B", "#2A2216"), "FAIL": ("#FF5A5F", "#2E1517"),
           "INVALID": ("#8C96A3", "#1A2027")}

FONT = '"Pretendard", "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans CJK KR", sans-serif'
MONO = '"IBM Plex Mono", "Consolas", "D2Coding", "Menlo", "DejaVu Sans Mono", monospace'
MONO_FAMILIES = ["IBM Plex Mono", "Consolas", "D2Coding", "Menlo", "DejaVu Sans Mono"]

QSS = f"""
* {{ font-family: {FONT}; font-size: 13px; color: {TEXT}; }}
QMainWindow, QWidget#page, QScrollArea, QScrollArea > QWidget > QWidget#page {{ background: {BG}; }}
QWidget#topbar {{ background: {CARD}; border-bottom: 1px solid {LINE}; }}
QWidget#topbar QPushButton {{ background: transparent; color: {TEXT2}; border: none; border-radius: 0;
    border-bottom: 2px solid transparent; padding: 11px 16px 9px 16px; font-size: 14px; font-weight: 600; }}
QWidget#topbar QPushButton:hover {{ color: {TEXT}; }}
QWidget#topbar QPushButton:checked {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QLabel#logo {{ font-family: {MONO}; font-size: 14px; font-weight: 600; color: {ACCENT}; padding: 0 18px 0 4px; }}
QLabel#topinfo {{ font-family: {MONO}; color: {MUTED}; font-size: 12px; }}
QLabel#dot {{ color: {OK}; font-size: 10px; }}
QFrame#card {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 4px; }}
QFrame#card QLabel {{ background: transparent; }}
QLabel#ph {{ color: {MUTED}; font-size: 11px; font-weight: 600; letter-spacing: 1px; }}
QFrame#sep {{ background: {LINE}; border: none; max-width: 1px; min-width: 1px; }}
QFrame#hsep {{ background: {LINE}; border: none; max-height: 1px; min-height: 1px; }}
QLabel#h1 {{ font-size: 20px; font-weight: 700; }}
QLabel#h2 {{ font-size: 16px; font-weight: 700; }}
QLabel#h3 {{ font-size: 14px; font-weight: 700; }}
QLabel#muted {{ color: {MUTED}; font-size: 12px; }}
QLabel#body2 {{ color: {TEXT2}; }}
QLabel#big {{ font-family: {MONO}; font-size: 22px; font-weight: 500; }}
QLabel#num {{ font-family: {MONO}; font-size: 15px; }}
QLabel#chip {{ background: transparent; color: {TEXT2}; border: 1px solid {LINE}; border-radius: 3px; padding: 1px 6px;
    font-family: {MONO}; font-size: 11px; }}
QPushButton {{ background: transparent; color: {TEXT}; border: 1px solid {LINE}; border-radius: 3px; padding: 7px 12px;
    font-weight: 600; font-size: 12px; }}
QPushButton:hover {{ background: {RAISED}; }}
QPushButton:disabled {{ color: {MUTED}; border-color: {LINE}; }}
QPushButton#primary {{ background: {ACCENT}; color: #111111; border: 1px solid {ACCENT}; }}
QPushButton#primary:hover {{ background: {ACCENT_HOVER}; }}
QPushButton#primary:disabled {{ background: {ACCENT_SOFT}; color: {MUTED}; border-color: {ACCENT_SOFT}; }}
QPushButton#secondary {{ background: transparent; color: {TEXT}; border: 1px solid #2C3540; }}
QPushButton#secondary:hover {{ background: {RAISED}; }}
QPushButton#link {{ background: transparent; color: {TEXT2}; border: none; padding: 6px 4px; font-weight: 500; }}
QPushButton#link:hover {{ color: {TEXT}; }}
QPushButton#seg {{ background: transparent; color: {MUTED}; border: none; border-bottom: 2px solid transparent;
    border-radius: 0; padding: 6px 10px; }}
QPushButton#seg:checked {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QLineEdit, QComboBox, QSpinBox, QDateEdit {{ background: {RAISED}; color: {TEXT}; border: 1px solid {LINE}; border-radius: 3px;
    padding: 6px 10px; }}
QLineEdit:focus, QComboBox:focus {{ border: 1px solid {ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QProgressBar {{ background: {LINE}; border: none; border-radius: 0; max-height: 3px; text-align: center; }}
QProgressBar::chunk {{ background: {ACCENT}; }}
QTableWidget {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 0; gridline-color: {LINE};
    alternate-background-color: #0F1318; selection-background-color: {ACCENT_SOFT}; outline: 0; }}
QTableWidget#plain {{ border: none; }}
QTableWidget::item {{ padding: 4px 8px; border-bottom: 1px solid {LINE}; }}
QHeaderView::section {{ background: {CARD}; color: {MUTED}; border: none; border-bottom: 1px solid {LINE};
    padding: 6px 8px; font-weight: 600; font-size: 11px; }}
QTabWidget::pane {{ border: 1px solid {LINE}; background: {BG}; }}
QTabBar::tab {{ background: transparent; color: {MUTED}; padding: 8px 14px; font-weight: 600; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QGroupBox {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 4px; margin-top: 18px; padding: 12px; font-weight: 600; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 6px; color: {MUTED}; font-size: 11px; }}
QPlainTextEdit {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 0; font-family: {MONO}; font-size: 12px; color: {TEXT2}; }}
QPlainTextEdit#log {{ border: none; }}
QStatusBar {{ background: {CARD}; color: {TEXT2}; border-top: 1px solid {LINE}; font-family: {MONO}; font-size: 12px; }}
QScrollBar:vertical {{ background: transparent; width: 8px; }}
QScrollBar::handle:vertical {{ background: #2A323C; border-radius: 4px; min-height: 30px; }}
QScrollBar:horizontal {{ background: transparent; height: 8px; }}
QScrollBar::handle:horizontal {{ background: #2A323C; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QMessageBox {{ background: {CARD}; }}
QComboBox QAbstractItemView {{ background: {CARD}; color: {TEXT}; selection-background-color: {ACCENT_SOFT}; border: 1px solid {LINE}; }}
QToolTip {{ background: {RAISED}; color: {TEXT}; border: 1px solid {LINE}; }}
QTableWidget QTableCornerButton::section {{ background: {CARD}; border: none; }}
"""


def mono_font(size: int = 12):
    """Fixed-width font for numbers in tables (QSS cannot reach table items)."""
    from PySide6.QtGui import QFont

    f = QFont()
    f.setFamilies(MONO_FAMILIES)
    f.setPointSizeF(size * 0.75)
    f.setStyleHint(QFont.Monospace)
    return f


def signed_color(x: float | None) -> str:
    if x is None or x != x or abs(x) < 1e-12:
        return TEXT
    return RISE if x > 0 else FALL


def apply(app) -> None:
    """Terminal palette regardless of the Windows theme, plus the stylesheet."""
    from PySide6.QtGui import QColor, QPalette

    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in ((QPalette.Window, BG), (QPalette.Base, CARD), (QPalette.AlternateBase, "#0F1318"),
                        (QPalette.Text, TEXT), (QPalette.WindowText, TEXT), (QPalette.Button, RAISED),
                        (QPalette.ButtonText, TEXT), (QPalette.Highlight, "#5A4520"), (QPalette.HighlightedText, TEXT),
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
