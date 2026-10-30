"""PySide6 main window. Talks only to an EngineController; never touches asyncio directly."""
from __future__ import annotations

from datetime import datetime
from html import escape
from typing import Optional

from PySide6.QtCore import QRect, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QGroupBox, QHBoxLayout, QHeaderView,
                               QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton, QSplitter, QTableWidget,
                               QTableWidgetItem, QTabWidget, QTextEdit, QVBoxLayout, QWidget)

from storage.face_store import FaceStore
from ui.qt_bridges import EngineController
from vision.tracker import UNKNOWN

BG, PANEL, BORDER = "#0f141a", "#171d26", "#263040"
TEXT, MUTED = "#d7dee8", "#8190a5"
ACCENT, WARN, DANGER = "#4cc38a", "#e0a43a", "#e5534b"

STYLE = f"""
QWidget {{ background: {BG}; color: {TEXT}; font-size: 13px; }}
QGroupBox {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px; margin-top: 14px; padding: 10px 8px 8px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; color: {MUTED}; background: {BG}; }}
QLabel#value {{ background: transparent; font-weight: 600; }}
QLabel#muted {{ background: transparent; color: {MUTED}; }}
QLabel {{ background: transparent; }}
QPushButton {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; padding: 6px 12px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:disabled {{ color: {MUTED}; }}
QPushButton#primary {{ background: {ACCENT}; color: #08130d; border: none; font-weight: 600; }}
QPushButton#primary:disabled {{ background: {PANEL}; color: {MUTED}; border: 1px solid {BORDER}; }}
QLineEdit {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; padding: 7px 9px; }}
QLineEdit:focus {{ border-color: {ACCENT}; }}
QListWidget, QTextEdit, QTableWidget {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; }}
QListWidget::item:selected, QTableWidget::item:selected {{ background: #20402f; }}
QHeaderView::section {{ background: {PANEL}; color: {MUTED}; border: none; padding: 4px; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: {BG}; color: {MUTED}; padding: 6px 14px; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {ACCENT}; }}
QProgressBar {{ background: {BG}; border: 1px solid {BORDER}; border-radius: 4px; height: 8px; text-align: center; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
QSplitter::handle {{ background: {BG}; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; }}
QCheckBox {{ background: transparent; }}
"""


class VideoWidget(QWidget):
    """Camera image with face boxes painted on top in widget coordinates (crisp at any size)."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(320, 240)
        self._image: Optional[QImage] = None
        self._tracks: list = []
        self._message = "Starting..."

    def set_frame(self, image: QImage, tracks: list) -> None:
        self._image, self._tracks, self._message = image, tracks, None
        self.update()

    def set_message(self, text: str) -> None:
        self._image, self._tracks, self._message = None, [], text
        self.update()

    @staticmethod
    def _style(t) -> tuple:
        if t.identity is None:
            return QColor(WARN), "verifying..."
        if t.identity == UNKNOWN:
            return QColor(DANGER), "Unknown"
        return QColor(ACCENT), f"{t.identity.replace('_', ' ')}  {t.score:.2f}"

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor("#0a0e13"))
        if self._image is None:
            p.setPen(QColor(MUTED))
            p.setFont(QFont(self.font().family(), 14))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message or "")
            return
        iw, ih = self._image.width(), self._image.height()
        scale = min(self.width() / iw, self.height() / ih)
        w, h = iw * scale, ih * scale
        x0, y0 = (self.width() - w) / 2, (self.height() - h) / 2
        p.drawImage(QRectF(x0, y0, w, h), self._image)

        p.setFont(QFont(self.font().family(), 11, QFont.Weight.DemiBold))
        fm = p.fontMetrics()
        for t in self._tracks:
            top, right, bottom, left = t.bbox
            box = QRectF(x0 + left * scale, y0 + top * scale, (right - left) * scale, (bottom - top) * scale)
            color, label = self._style(t)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.setPen(QPen(color, 3))
            p.drawRoundedRect(box, 6, 6)
            pill_w, pill_h = fm.horizontalAdvance(label) + 18, fm.height() + 8
            y = box.bottom() + 5 if box.bottom() + 5 + pill_h < self.height() else box.top() - pill_h - 5
            pill = QRectF(box.left(), y, pill_w, pill_h)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            p.drawRoundedRect(pill, 5, 5)
            p.setPen(QColor("#08130d"))
            p.drawText(pill, Qt.AlignmentFlag.AlignCenter, label)


def _value_label(text: str = "-") -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("value")
    return lab


def _muted_label(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setObjectName("muted")
    return lab


class MainWindow(QMainWindow):
    def __init__(self, controller: EngineController) -> None:
        super().__init__()
        self.ctl = controller
        self.setWindowTitle("Aether")
        self.setMinimumSize(860, 520)
        self.resize(1200, 740)
        self.setStyleSheet(STYLE)
        self._has_camera = self._has_mic = False
        self._stopped = False

        self.video = VideoWidget()
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.video)
        splitter.addWidget(self._build_side_panel())
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([740, 440])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Starting...")
        screen = QApplication.primaryScreen()
        if screen is not None:
            self.fit_to_screen(screen.availableGeometry())

        # Connect to bound methods of this widget (not lambdas): Qt then always delivers the call on the
        # GUI thread, even though the signals are emitted from the engine thread.
        b = self.ctl.bridge
        b.frame_available.connect(self._on_frame)
        b.reply.connect(self._on_reply)
        b.heard.connect(self._on_heard)
        b.voice_state.connect(self._on_voice_state)
        b.people_changed.connect(self.refresh_people)
        b.activity.connect(self._on_activity)
        b.error.connect(self._on_error)
        b.status.connect(self.statusBar().showMessage)
        b.shutdown_requested.connect(self.close)
        self.ctl.started.connect(self._on_started)

        self._poll = QTimer(self)
        self._poll.timeout.connect(self._poll_state)
        self._poll.start(100)
        self._last_ms = 0.0
        self._brightness = 255.0
        self._voice_text = "-"

    def fit_to_screen(self, avail: QRect) -> None:
        """Never taller or wider than the usable screen (a 1366x768 laptop minus the taskbar)."""
        self.resize(min(1200, int(avail.width() * 0.92)), min(740, int(avail.height() * 0.92)))
        self.move(avail.center() - self.rect().center())

    # ---- layout ----------------------------------------------------------
    def _build_side_panel(self) -> QWidget:
        side = QWidget()
        lay = QVBoxLayout(side)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.addWidget(self._build_status_box())
        lay.addWidget(self._build_people_box())
        lay.addWidget(self._build_tabs(), 1)
        lay.addLayout(self._build_input_row())
        return side

    def _build_status_box(self) -> QGroupBox:
        box = QGroupBox("System")
        grid = QVBoxLayout(box)
        row = QHBoxLayout()
        self.lbl_fps, self.lbl_ms, self.lbl_voice = _value_label(), _value_label(), _value_label("-")
        for caption, lab in (("Camera", self.lbl_fps), ("Detect", self.lbl_ms), ("Voice", self.lbl_voice)):
            col = QVBoxLayout()
            col.addWidget(_muted_label(caption))
            col.addWidget(lab)
            row.addLayout(col)
        grid.addLayout(row)
        self.mic_bar = QProgressBar()
        self.mic_bar.setRange(0, 100)
        self.mic_bar.setTextVisible(False)
        grid.addWidget(self.mic_bar)
        toggles = QHBoxLayout()
        self.chk_listen = QCheckBox("Listening")
        self.chk_listen.setChecked(True)
        self.chk_listen.toggled.connect(self.ctl.set_listening)
        self.btn_talk = QPushButton("Talk (F9)")
        self.btn_talk.setToolTip("Push to talk: the next thing you say is a command, no wake word needed")
        self.btn_talk.clicked.connect(self._talk)
        self.btn_talk.setEnabled(False)
        QShortcut(QKeySequence("F9"), self, activated=self._talk)
        self.chk_mute = QCheckBox("Mute voice replies")
        self.chk_mute.toggled.connect(self.ctl.set_muted)
        toggles.addWidget(self.btn_talk)
        toggles.addWidget(self.chk_listen)
        toggles.addWidget(self.chk_mute)
        toggles.addStretch()
        grid.addLayout(toggles)
        self.lbl_hint = QLabel("Dim light: the camera slows down and recognition is less accurate.")
        self.lbl_hint.setStyleSheet(f"color: {WARN}; background: transparent;")
        self.lbl_hint.setWordWrap(True)
        self.lbl_hint.hide()
        grid.addWidget(self.lbl_hint)
        return box

    def _build_people_box(self) -> QGroupBox:
        box = QGroupBox("People")
        lay = QVBoxLayout(box)
        self.people = QListWidget()
        self.people.setMaximumHeight(90)
        lay.addWidget(self.people)
        self.enroll_bar = QProgressBar()
        self.enroll_bar.setTextVisible(False)
        self.enroll_lbl = _muted_label("")
        self.enroll_bar.hide()
        self.enroll_lbl.hide()
        lay.addWidget(self.enroll_lbl)
        lay.addWidget(self.enroll_bar)
        self.lbl_overlap = QLabel("")
        self.lbl_overlap.setStyleSheet(f"color: {WARN}; background: transparent;")
        self.lbl_overlap.setWordWrap(True)
        self.lbl_overlap.hide()
        lay.addWidget(self.lbl_overlap)
        row = QHBoxLayout()
        self.btn_enroll = QPushButton("Enroll new person...")
        self.btn_enroll.setObjectName("primary")
        self.btn_enroll.clicked.connect(self._enroll)
        self.btn_add = QPushButton("Add samples")
        self.btn_add.setToolTip("Capture more views of the selected person")
        self.btn_add.clicked.connect(self._add_samples)
        self.btn_forget = QPushButton("Forget selected")
        self.btn_forget.clicked.connect(self._forget)
        row.addWidget(self.btn_enroll)
        row.addWidget(self.btn_add)
        row.addWidget(self.btn_forget)
        lay.addLayout(row)
        return box

    def _build_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.setMinimumHeight(110)
        self.chat = QTextEdit()
        self.chat.setReadOnly(True)
        tabs.addTab(self.chat, "Conversation")
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Time", "Event", "Detail"])
        for col in (0, 1):
            self.table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        tabs.addTab(self.table, "Activity")
        return tabs

    def _build_input_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Type a command, e.g. who is at the camera (no wake word needed)")
        self.input.returnPressed.connect(self._send)
        btn = QPushButton("Send")
        btn.clicked.connect(self._send)
        row.addWidget(self.input, 1)
        row.addWidget(btn)
        return row

    # ---- slots -----------------------------------------------------------
    def _on_started(self, warnings: list, has_camera: bool, has_mic: bool) -> None:
        self._has_camera, self._has_mic = has_camera, has_mic
        self.statusBar().showMessage("Ready")
        for w in warnings:
            self._log("System", w, WARN)
        if not has_camera:
            self.video.set_message("Camera is off")
        elif self.video._image is None:
            self.video.set_message("Waiting for the camera...")
        self._voice_text = "Listening" if has_mic else "Mic off"
        self.lbl_voice.setText(self._voice_text)
        self.chk_listen.setEnabled(has_mic)
        self.btn_talk.setEnabled(has_mic)
        self.refresh_people()
        self.refresh_activity()
        self._log("Aether", f'Ready. Type a command below, say "{self.ctl.wake_name}, help", or press F9 and talk.', ACCENT)

    def _on_frame(self) -> None:
        item = self.ctl.bridge.take_frame()
        if item is None:
            return
        image, info = item
        self._last_ms = info.inference_ms
        self._brightness = info.brightness
        self.video.set_frame(image, info.tracks)

    def _on_reply(self, text: str) -> None:
        self._log("Aether", text, ACCENT)

    def _on_error(self, message: str) -> None:
        self._log("System", message, DANGER)

    def _on_activity(self) -> None:
        QTimer.singleShot(250, self.refresh_activity)  # give the audit row a moment to be written

    def _on_heard(self, text: str, addressed: bool) -> None:
        if addressed:
            self._log("You", text, TEXT)
        else:
            self._log("Heard", f"{text} (ignored: no wake word)", MUTED)

    def _on_voice_state(self, state: str) -> None:
        if self._has_mic and self.chk_listen.isChecked():
            self._voice_text = {"hearing": "Hearing you...", "transcribing": "Transcribing..."}.get(state, "Listening")

    def _talk(self) -> None:
        if self._has_mic:
            self.ctl.listen_now()

    def _poll_state(self) -> None:
        if self._stopped:
            return
        self.lbl_fps.setText(f"{self.ctl.fps:.0f} FPS" if self._has_camera else "off")
        self.lbl_ms.setText(f"{self._last_ms:.0f} ms" if self._has_camera else "-")
        self.lbl_hint.setVisible(self._has_camera and self._brightness < 60)
        att = self.ctl.attention  # while listening without a wake word, say so: the user must know
        if att is not None and self._has_mic:
            kind, left = att
            self.lbl_voice.setText(f"Your turn ({left:.0f}s)" if kind == "talk" else f"Follow-up ({left:.0f}s)")
            self.lbl_voice.setStyleSheet(f"color: {ACCENT}; background: transparent; font-weight: 600;")
        elif self._has_mic:
            self.lbl_voice.setText(self._voice_text)
            self.lbl_voice.setStyleSheet("")
        self.mic_bar.setValue(int(max(0.0, min(100.0, (self.ctl.mic_level_db + 60) * 100 / 60))))
        prog = self.ctl.enrollment_progress
        busy = prog is not None
        self.enroll_bar.setVisible(busy)
        self.enroll_lbl.setVisible(busy)
        self.btn_enroll.setEnabled(not busy)
        self.btn_add.setEnabled(not busy and self._selected_person() is not None)
        if busy:
            name, got, need = prog
            self.enroll_bar.setRange(0, need)
            self.enroll_bar.setValue(got)
            self.enroll_lbl.setText(f"Enrolling {name.replace('_', ' ')} ({got}/{need}): {self.ctl.enrollment_hint}")

    def _send(self) -> None:
        text = self.input.text().strip()
        if text:
            self.input.clear()
            self.ctl.send_text(text)

    def _enroll(self) -> None:
        name, ok = QInputDialog.getText(self, "Enroll new person", "Name (letters, digits, - or _):")
        name = name.strip().replace(" ", "_")
        if not ok or not name:
            return
        if not FaceStore.valid_name(name):
            QMessageBox.warning(self, "Invalid name", "Use only letters, digits, '-' and '_'.")
            return
        fut = self.ctl.begin_enrollment(name)
        fut.add_done_callback(lambda f: self.ctl.bridge.error.emit(str(f.exception())) if f.exception() else None)

    def _selected_person(self) -> Optional[str]:
        item = self.people.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item is not None else None

    def _add_samples(self) -> None:
        name = self._selected_person()
        if name is None:
            return
        fut = self.ctl.begin_enrollment(name)  # same name: adds more views to that person
        fut.add_done_callback(lambda f: self.ctl.bridge.error.emit(str(f.exception())) if f.exception() else None)

    def _forget(self) -> None:
        name = self._selected_person()
        if name is None:
            return
        answer = QMessageBox.question(self, "Forget person",
                                      f"Delete all stored face data for {name.replace('_', ' ')}?")
        if answer == QMessageBox.StandardButton.Yes:
            self.ctl.forget(name)

    # ---- views -----------------------------------------------------------
    def refresh_people(self) -> None:
        if self._stopped:  # a timer that was already scheduled can fire after the window closed
            return
        keep = self.people.currentItem().data(Qt.ItemDataRole.UserRole) if self.people.currentItem() else None
        self.people.clear()
        for name, n in sorted(self.ctl.enrolled().items()):
            item = QListWidgetItem(f"{name.replace('_', ' ')}   ({n} samples)")
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.people.addItem(item)
            if name == keep:
                self.people.setCurrentItem(item)
        if self.people.count() == 0:
            self.people.addItem(QListWidgetItem("Nobody enrolled yet"))
            self.people.item(0).setFlags(Qt.ItemFlag.NoItemFlags)
        self.btn_forget.setEnabled(self.people.count() > 0 and self.people.item(0).data(Qt.ItemDataRole.UserRole) is not None)
        overlaps = self.ctl.identity_overlaps()
        self.lbl_overlap.setText("\n".join(overlaps))
        self.lbl_overlap.setVisible(bool(overlaps))

    def refresh_activity(self) -> None:
        if self._stopped:
            return
        rows = self.ctl.recent_activity(100)
        self.table.setRowCount(len(rows))
        for r, (ts, kind, detail) in enumerate(rows):
            for c, value in enumerate((ts[11:19], kind.replace("_", " "), detail)):
                self.table.setItem(r, c, QTableWidgetItem(value))

    def _log(self, who: str, text: str, color: str) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        self.chat.append(f'<span style="color:{MUTED}">{now}</span> '
                         f'<b style="color:{color}">{escape(who)}</b>&nbsp; '
                         f'<span style="color:{TEXT if who != "Heard" else MUTED}">{escape(text)}</span>')

    # ---- closing ---------------------------------------------------------
    def closeEvent(self, event) -> None:
        if not self._stopped:
            self._stopped = True
            self._poll.stop()
            self.statusBar().showMessage("Shutting down...")
            self.ctl.stop()
        event.accept()
