# main.py (Revised - Added Save SRT functionality)

import sys
import os
import shutil
import time
from pathlib import Path
import urllib.request

# Attempt to import PySide6 components
try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
        QPushButton, QTabWidget, QSplitter, QMenuBar, QToolBar, QFileDialog,
        QMessageBox, QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QSizePolicy,
        QDialog, QDialogButtonBox, QFormLayout, QStyleFactory, QStatusBar, QGroupBox,
        QRadioButton, QCheckBox, QToolButton, QCommandLinkButton, QDateTimeEdit,
        QSlider, QScrollBar, QDial, QDial, QProgressBar, QGridLayout, QMenu, QInputDialog,
        QPlainTextEdit
    )
    from PySide6.QtGui import QAction, QKeySequence, QTextCursor, QColor, QTextDocument, QFont, QTextCharFormat, QBrush
    from PySide6.QtCore import Qt, Slot, QSize, QSettings, QFile, QTextStream, QDateTime, QTimer, QUrl, QPoint
    from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("ERROR: PySide6 is not available. Please install: pip install PySide6 PySide6-Addons PySide6.QtMultimedia", file=sys.stderr)
    sys.exit(1)

# Attempt to import model.py
try:
    import model

    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print(
        "ERROR: model.py not found. Ensure model.py is in the same directory.",
        file=sys.stderr,
    )
    pass

# --- Constants ---
APP_NAME = "Parakeet_GUI"
APP_VERSION = "0.1.6"  # Increment version for save functionality
SETTINGS_ORG = "ThereIsNoSource.org"
SETTINGS_APP = APP_NAME
DEFAULT_WINDOW_SIZE = QSize(1024, 768)
DEFAULT_FONT_SIZE = 11
DEFAULT_HIGHLIGHT_COLOR = "#666688"  # light blue
DEFAULT_HIGHLIGHT_DURATION_MS = 50  # How often the timer fires for simulation


# --- Main Application Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Use IniFormat for easier inspection if needed
        self.settings = QSettings(
            str(Path.cwd() / "settings.ini"), QSettings.Format.IniFormat
        )

        self.timed_text_data = None  # Structured data from model.py
        self.current_sim_index = 0  # Index for current unit in flat list

        self.simulation_timer = QTimer(self)
        self.simulation_timer.timeout.connect(
            self._update_highlight_from_timer
        )
        self.sim_start_time_real = (
            0  # Real time when timer-based simulation started
        )
        self.sim_start_time_data = (
            0  # Data time (in seconds) corresponding to sim_start_time_real
        )
        self.simulation_speed_multiplier = 1.0

        self.media_player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.media_player.setAudioOutput(self.audio_output)
        self.media_player.playbackStateChanged.connect(
            self._media_player_state_changed
        )
        self.media_player.positionChanged.connect(
            self._update_highlight_from_playback
        )  # Connect for playback sync
        self.current_audio_path = (
            None  # Path to the currently loaded audio file
        )

        self.highlight_format = QTextCharFormat()
        self.highlight_format.setBackground(QColor(DEFAULT_HIGHLIGHT_COLOR))
        self.default_format = QTextCharFormat()
        # Explicitly set background brush to NoBrush to remove highlighting
        self.default_format.setBackground(QBrush(Qt.BrushStyle.NoBrush))

        # Store the previous highlight range for efficient clearing
        self._previous_highlight_range = (0, 0)  # (start_pos, end_pos)

        # Map to store character positions for highlighting for *all* levels
        # Structure: {(sentence_idx, word_idx, char_idx): (start_char_pos, end_char_pos), ...}
        # (-1, -1, -1) means sentence key, (s_idx, w_idx, -1) for word keys, (s_idx, w_idx, c_idx) for char keys.
        self._text_pos_maps = {"Sentence": {}, "Word": {}, "Character": {}}
        self._ignore_playback_updates = False  # Flag to prevent position reset on stop

        self._init_ui()
        self._load_settings()
        self._apply_current_style_and_theme()
        self._check_dependencies()

        self._update_ui_state()  # Set initial button states

    def _init_ui(self):
        """Creates the user interface elements."""
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setGeometry(
            100, 100, DEFAULT_WINDOW_SIZE.width(), DEFAULT_WINDOW_SIZE.height()
        )

        # Menu Bar
        self._create_menu_bar()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        top_control_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(top_control_splitter)

        file_options_group = self._create_file_options_group()
        top_control_splitter.addWidget(file_options_group)

        sim_controls_group = self._create_sim_controls_group()
        top_control_splitter.addWidget(sim_controls_group)

        # Initial split ratios (adjust as needed)
        top_control_splitter.setSizes(
            [int(self.width() * 0.4), int(self.width() * 0.6)]
        )

        # Use a vertical splitter for text display and log
        bottom_splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(bottom_splitter, 1)  # Give it stretch factor 1

        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setPlaceholderText(
            "Load an audio file or SRT to display transcript here..."
        )
        self.text_display.setAcceptRichText(
            False
        )  # Crucial for consistent plain text mapping
        bottom_splitter.addWidget(self.text_display)

        self.log_display = QPlainTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setPlaceholderText(
            "Application Log / Detailed Errors will appear here..."
        )
        self.log_display.setMaximumHeight(150)  # Initial max height
        bottom_splitter.addWidget(self.log_display)

        # Set initial sizes for bottom splitter (e.g., 70% text, 30% log)
        bottom_splitter.setSizes(
            [int(self.height() * 0.7), int(self.height() * 0.3)]
        )

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Connect selection changed for click-to-jump functionality
        # Also handle cursor position changes (which trigger selectionChanged with empty selection)
        self.text_display.selectionChanged.connect(
            self._on_text_selection_changed
        )

    def _create_menu_bar(self):
        """Creates the application's menu bar."""
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("&File")

        # Open File Action
        open_action = QAction("&Open File...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.setStatusTip("Open an audio or SRT file")
        open_action.triggered.connect(self._browse_file)
        file_menu.addAction(open_action)

        # Save SRT Action
        self.save_srt_action = QAction("&Save SRT...", self)
        self.save_srt_action.setShortcut(QKeySequence.StandardKey.Save)
        self.save_srt_action.setStatusTip(
            "Save the current transcription as an SRT file"
        )
        self.save_srt_action.triggered.connect(self._save_srt_file)
        self.save_srt_action.setEnabled(False)  # Disabled by default
        file_menu.addAction(self.save_srt_action)

        # Separator
        file_menu.addSeparator()

        # Exit Action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)  # Connect to close()
        file_menu.addAction(exit_action)

        # Help Menu (Optional, Placeholder)
        # help_menu = menu_bar.addMenu("&Help")
        # about_action = QAction("&About", self)
        # about_action.triggered.connect(self._show_about_dialog)
        # help_menu.addAction(about_action)

    def _create_file_options_group(self):
        """Creates the group box for file selection and transcription options."""
        group_box = QGroupBox("File && Options")
        layout = QGridLayout(group_box)

        layout.addWidget(QLabel("Audio/SRT File:"), 0, 0)
        self.file_path_lineedit = QLineEdit()
        self.file_path_lineedit.setReadOnly(True)
        self.file_path_lineedit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        layout.addWidget(self.file_path_lineedit, 0, 1)
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self._browse_file)
        layout.addWidget(self.browse_button, 0, 2)

        layout.addWidget(QLabel("Segment Length (sec):"), 1, 0)
        self.segment_length_spinbox = QSpinBox()
        self.segment_length_spinbox.setRange(1, 600)
        self.segment_length_spinbox.setValue(60)
        layout.addWidget(self.segment_length_spinbox, 1, 1)

        self.transcribe_button = QPushButton("Transcribe / Load File")
        self.transcribe_button.clicked.connect(self._handle_transcribe_or_load)
        # Initial state set by _update_ui_state
        layout.addWidget(
            self.transcribe_button,
            2,
            1,
            1,
            1,
            alignment=Qt.AlignmentFlag.AlignRight,
        )

        # Add button for Style/Theme settings
        self.settings_button = QPushButton("Settings...")
        self.settings_button.clicked.connect(self._show_settings_dialog)
        layout.addWidget(
            self.settings_button,
            2,
            0,
            1,
            1,
            alignment=Qt.AlignmentFlag.AlignLeft,
        )

        layout.setRowStretch(3, 1)
        layout.setColumnStretch(1, 1)

        return group_box

    def _create_sim_controls_group(self):
        """Creates the group box for simulation settings and controls."""
        group_box = QGroupBox("Simulation && Playback")
        layout = QGridLayout(group_box)

        layout.addWidget(QLabel("Highlight Level:"), 0, 0)
        self.highlight_level_combo = QComboBox()
        self.highlight_level_combo.addItems(["Word", "Sentence", "Character"])
        self.highlight_level_combo.setCurrentText("Word")
        self.highlight_level_combo.currentTextChanged.connect(
            self._on_highlight_level_changed
        )
        layout.addWidget(self.highlight_level_combo, 0, 1)

        layout.addWidget(QLabel("Sim Speed Multiplier:"), 1, 0)
        self.sim_speed_spinbox = QDoubleSpinBox()
        self.sim_speed_spinbox.setRange(0.1, 5.0)
        self.sim_speed_spinbox.setSingleStep(0.1)
        self.sim_speed_spinbox.setValue(1.0)
        self.sim_speed_spinbox.valueChanged.connect(self._update_sim_speed)
        layout.addWidget(self.sim_speed_spinbox, 1, 1)

        self.play_button = QPushButton("Play Audio")
        self.play_button.clicked.connect(self._handle_play)

        self.play_section_button = QPushButton("Play Section")
        self.play_section_button.clicked.connect(self._handle_play_section)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._handle_stop)

        self.prev_button = QPushButton("Previous Unit")
        self.prev_button.clicked.connect(lambda: self._navigate_unit(-1))

        self.next_button = QPushButton("Next Unit")
        self.next_button.clicked.connect(lambda: self._navigate_unit(1))

        # Initial states set by _update_ui_state
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.play_section_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch()  # Push navigation buttons to the right
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)

        layout.addLayout(button_layout, 2, 0, 1, 2)

        layout.setRowStretch(3, 1)
        layout.setColumnStretch(1, 1)

        return group_box

    def _update_ui_state(self):
        """Updates the enabled/disabled state of controls based on application state."""
        has_file_path = bool(self.file_path_lineedit.text())
        has_timed_data = (
            self.timed_text_data is not None and len(self.timed_text_data) > 0
        )
        has_audio_path = (
            self.current_audio_path is not None
            and Path(self.current_audio_path).exists()
        )

        # File/Option Controls
        self.transcribe_button.setEnabled(has_file_path and MODEL_AVAILABLE)
        # settings button is always enabled

        # Playback/Simulation Controls
        self.play_button.setEnabled(has_timed_data and has_audio_path)
        self.play_section_button.setEnabled(
            has_timed_data
        )  # Can play section even without audio
        self.stop_button.setEnabled(has_timed_data)
        self.prev_button.setEnabled(has_timed_data)
        self.next_button.setEnabled(has_timed_data)
        self.highlight_level_combo.setEnabled(has_timed_data)
        self.sim_speed_spinbox.setEnabled(
            has_timed_data
        )  # Speed is for simulation

        # Menu Actions
        self.save_srt_action.setEnabled(has_timed_data and MODEL_AVAILABLE)

    def _check_dependencies(self):
        """Performs initial checks for external dependencies like FFmpeg."""
        self.log_display.appendPlainText("Performing dependency checks...")
        all_ok_for_transcription = True  # Assume OK unless proven otherwise
        model_present = MODEL_AVAILABLE  # Check if model.py was imported

        if model_present:
            try:
                if model.check_ffmpeg_available():
                    self.log_display.appendPlainText("- FFmpeg found in PATH.")
                else:
                    self.log_display.appendPlainText(
                        "WARNING: FFmpeg not found in PATH. Audio processing will fail for non-WAV files or if pydub needs it."
                    )
                    QMessageBox.warning(
                        self,
                        "Dependency Missing",
                        "FFmpeg not found in system PATH.\nAudio transcription requires FFmpeg (especially for MP3, etc.). Please install it and add its 'bin' directory to PATH.",
                    )
                    all_ok_for_transcription = False
            except Exception as e:
                self.log_display.appendPlainText(f"ERROR checking FFmpeg: {e}")
                QMessageBox.critical(
                    self,
                    "Dependency Check Error",
                    f"An error occurred while checking FFmpeg: {e}",
                )
                all_ok_for_transcription = False

            if model.NEMO_AVAILABLE:
                self.log_display.appendPlainText(
                    "- NeMo toolkit and PyTorch imported."
                )
                if hasattr(model, "torch") and model.torch.cuda.is_available():
                    self.log_display.appendPlainText(
                        f"- CUDA detected. GPU: {model.torch.cuda.get_device_name(0)}"
                    )
                else:
                    self.log_display.appendPlainText(
                        "WARNING: CUDA not detected or available. NeMo Parakeet model requires CUDA. Transcription will fail."
                    )
                    all_ok_for_transcription = (
                        False  # Transcription will fail without CUDA
                    )
            else:
                self.log_display.appendPlainText(
                    "ERROR: NeMo toolkit or PyTorch not available. Transcription will not work."
                )
                QMessageBox.critical(
                    self,
                    "Dependency Missing",
                    "NeMo toolkit or PyTorch is not installed or importable.\nTranscription requires these libraries and a CUDA-enabled GPU setup.\nPlease check your environment.",
                )
                all_ok_for_transcription = False

            if model.PYDUB_AVAILABLE:
                self.log_display.appendPlainText("- pydub imported.")
            else:
                self.log_display.appendPlainText(
                    "ERROR: pydub not available. Audio processing will not work."
                )
                QMessageBox.critical(
                    self,
                    "Dependency Missing",
                    "pydub is not installed or importable.\nAudio processing requires this library.\nPlease install it (`pip install pydub`).",
                )
                all_ok_for_transcription = False

        else:
            self.log_display.appendPlainText(
                "FATAL ERROR: model.py not available. Cannot perform dependency checks or transcription."
            )
            QMessageBox.critical(
                self,
                "Fatal Error",
                "model.py could not be imported. Application cannot run its core functions.",
            )
            # model_present is False, transcription and save buttons will be disabled by _update_ui_state

        self.log_display.appendPlainText("Dependency checks complete.")
        if model_present and not all_ok_for_transcription:
            self.log_display.appendPlainText(
                "WARNING: Transcription functionality may be limited or unavailable due to missing dependencies or CUDA."
            )

        self._update_ui_state()  # Update button states based on checks

    @Slot()
    def _browse_file(self):
        """Opens a file dialog to select an audio or SRT file."""
        file_filter = "Media Files (*.mp3 *.wav *.aac *.flac *.ogg *.m4a *.srt);;Audio Files (*.mp3 *.wav *.aac *.flac *.ogg *.m4a);;SRT Files (*.srt);;All Files (*)"
        # Get last used directory from settings if available
        last_dir = self.settings.value(
            "last_file_dir", str(Path.home()), type=str
        )

        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio or SRT File", last_dir, file_filter
        )

        if file_path:
            self.file_path_lineedit.setText(file_path)
            # Save the directory
            self.settings.setValue(
                "last_file_dir", str(Path(file_path).parent)
            )

            # Reset application state
            self._handle_stop()  # Stop any ongoing processes
            self.current_audio_path = (
                None  # Reset audio path until confirmed or transcribed
            )
            self.timed_text_data = None
            self._clear_position_maps()
            self.current_sim_index = 0
            self._previous_highlight_range = (0, 0)  # Reset highlight range

            self.text_display.clear()  # Clear text display
            self.text_display.setPlaceholderText("Processing file...")

            self.log_display.clear()  # Clear log display
            self.log_display.appendPlainText(
                "New file selected. Ready to process."
            )

            self.status_bar.showMessage(
                f"File selected: {Path(file_path).name}", 3000
            )
            self._update_ui_state()  # Update button states

    def _clear_position_maps(self):
        """Clears all stored text position maps."""
        self._text_pos_maps = {"Sentence": {}, "Word": {}, "Character": {}}

    @Slot()
    def _handle_transcribe_or_load(self):
        """Handles the button click to transcribe audio or load SRT."""
        file_path = self.file_path_lineedit.text()
        if not file_path:
            QMessageBox.warning(
                self,
                "No File Selected",
                "Please browse and select an audio or SRT file first.",
            )
            return

        if not MODEL_AVAILABLE:
            QMessageBox.critical(
                self,
                "Fatal Error",
                "model.py could not be imported. Cannot process files.",
            )
            self._update_ui_state()  # Ensure buttons are disabled
            return

        file_path_obj = Path(file_path)

        self._handle_stop()  # Stop any running processes first
        self._update_ui_state()  # Disable controls during processing
        self.text_display.clear()
        self.text_display.setPlaceholderText("Processing file...")
        self.timed_text_data = None
        self._clear_position_maps()
        self.current_sim_index = 0
        self._previous_highlight_range = (0, 0)
        self.status_bar.showMessage("Processing file...", 0)
        # Preserve previous log messages but add new ones
        self.log_display.appendPlainText(
            f"\n--- Processing file: {file_path_obj.name} ---"
        )
        QApplication.processEvents()  # Update GUI

        if file_path_obj.suffix.lower() == ".srt":
            self.current_audio_path = None  # Assume no audio unless explicitly provided by user later
            self.log_display.appendPlainText(
                "Detected .srt file. Attempting to load..."
            )
            self._load_srt_file(str(file_path_obj))
        else:
            # For audio files, first check for an associated SRT
            srt_path_candidate = file_path_obj.with_suffix(".srt")
            if srt_path_candidate.exists():
                self.log_display.appendPlainText(
                    f"Found associated SRT file: {srt_path_candidate.name}. Attempting to load SRT..."
                )
                # Pass the associated audio path to the SRT loader
                self._load_srt_file(
                    str(srt_path_candidate), audio_path=str(file_path_obj)
                )
                # _load_srt_file will attempt transcription if SRT load fails or returns no data AND audio_path was provided

            else:
                self.log_display.appendPlainText(
                    f"No associated SRT found for {file_path_obj.name}. Starting transcription..."
                )
                # Pass the audio path for transcription
                self._perform_transcription(str(file_path_obj))

        self._update_ui_state()  # Update button states after processing attempt

    def _load_srt_file(self, srt_path: str, audio_path: str = None):
        """Loads timed text from an SRT file."""
        if not MODEL_AVAILABLE:
            return  # Should be checked by caller

        try:
            self.timed_text_data = model.load_srt_timed_text(srt_path)

            if self.timed_text_data:
                self.current_audio_path = (
                    audio_path  # Keep audio path if provided
                )
                if self.current_audio_path:
                    self.log_display.appendPlainText(
                        f"Associated audio path set: {Path(self.current_audio_path).name}"
                    )
                    # Check if audio file exists for playback
                    if not Path(self.current_audio_path).exists():
                        self.log_display.appendPlainText(
                            "WARNING: Associated audio file not found for playback."
                        )
                        self.current_audio_path = (
                            None  # Disable playback if file doesn't exist
                        )
                    # If audio path *was* provided but the file doesn't exist, should we try transcribing?
                    # No, let's only attempt transcription if SRT load failed or returned no data.

                else:
                    self.log_display.appendPlainText(
                        "No associated audio path provided for SRT. Playback disabled."
                    )

                self._display_timed_text()  # Display and build maps

                self.status_bar.showMessage(
                    f"SRT loaded: {Path(srt_path).name}", 5000
                )
                self.log_display.appendPlainText("SRT loading successful.")

            else:
                # SRT loaded but empty or invalid structure returned
                self.log_display.appendPlainText(
                    f"WARNING: SRT loaded successfully but returned no data."
                )
                if audio_path:
                    self.log_display.appendPlainText(
                        f"Attempting transcription for {Path(audio_path).name} instead."
                    )
                    # Clear current state before transcription attempt
                    self.timed_text_data = None
                    self._clear_position_maps()
                    self.current_sim_index = 0
                    self._previous_highlight_range = (0, 0)
                    self.text_display.clear()
                    self.text_display.setPlaceholderText(
                        "Starting transcription..."
                    )

                    self._perform_transcription(
                        audio_path
                    )  # Attempt transcription if audio path available
                else:
                    self.status_bar.showMessage(
                        f"SRT loaded but empty. No audio provided to transcribe.",
                        5000,
                    )
                    self.log_display.appendPlainText(
                        "SRT loading successful but returned no data. No associated audio to transcribe."
                    )

        except (FileNotFoundError, model.SRTParseError) as e:
            error_msg = f"Failed to load SRT file: {e}"
            self.log_display.appendPlainText(f"ERROR: {error_msg}")
            QMessageBox.critical(self, "SRT Load Error", error_msg)
            self.status_bar.showMessage("SRT loading failed.", 5000)
            self.text_display.setPlaceholderText("Failed to load SRT.")
        except Exception as e:
            error_msg = f"An unexpected error occurred during SRT loading: {e}"
            self.log_display.appendPlainText(f"ERROR: {error_msg}")
            import traceback

            self.log_display.appendPlainText(traceback.format_exc())
            QMessageBox.critical(self, "SRT Load Error", error_msg)
            self.status_bar.showMessage("SRT loading failed.", 5000)
            self.text_display.setPlaceholderText("Failed to load SRT.")
        finally:
            self._update_ui_state()  # Always update UI state after load attempt

    def _perform_transcription(self, audio_path: str):
        """Initiates the transcription process in model.py."""
        if not MODEL_AVAILABLE:
            return  # Should be checked by caller

        segment_length = self.segment_length_spinbox.value()
        self.current_audio_path = audio_path  # Set audio path for playback

        self.status_bar.showMessage(
            "Transcribing audio (GUI might become unresponsive)...", 0
        )
        self.log_display.appendPlainText(
            f"Calling model.transcribe_audio_and_time('{Path(audio_path).name}', {segment_length})..."
        )
        QApplication.processEvents()  # Update GUI before long task

        try:
            self.timed_text_data = model.transcribe_audio_and_time(
                audio_path, segment_length
            )

            if self.timed_text_data:
                self._display_timed_text()  # This also builds the position maps
                self.status_bar.showMessage("Transcription complete.", 5000)
                self.log_display.appendPlainText("Transcription successful.")

            else:
                error_msg = (
                    "Transcription completed but returned no timed data."
                )
                self.log_display.appendPlainText(f"WARNING: {error_msg}")
                QMessageBox.warning(self, "Transcription Warning", error_msg)
                self.status_bar.showMessage(
                    "Transcription finished with no data.", 5000
                )
                self.text_display.setPlaceholderText(
                    "Transcription resulted in no data."
                )

        except (
            model.FFmpegNotFoundError,
            model.InvalidAudioFileError,
            model.NemoInitializationError,
            model.TranscriptionError,
        ) as e:
            error_msg = f"Transcription failed: {e}"
            self.log_display.appendPlainText(f"ERROR: {error_msg}")
            # Provide specific dependency messages in critical dialog
            if isinstance(e, model.FFmpegNotFoundError):
                QMessageBox.critical(
                    self,
                    "Transcription Error: FFmpeg",
                    f"Transcription failed because FFmpeg was not found:\n{e}",
                )
            elif isinstance(e, model.NemoInitializationError):
                QMessageBox.critical(
                    self,
                    "Transcription Error: NeMo/CUDA",
                    f"Transcription failed due to a NeMo/CUDA issue:\n{e}",
                )
            elif isinstance(e, model.InvalidAudioFileError):
                QMessageBox.critical(
                    self,
                    "Transcription Error: Audio File",
                    f"Transcription failed due to an audio file issue:\n{e}",
                )
            else:
                QMessageBox.critical(self, "Transcription Error", error_msg)

            self.status_bar.showMessage("Transcription failed.", 5000)
            self.text_display.setPlaceholderText("Transcription failed.")

        except Exception as e:
            error_msg = (
                f"An unexpected error occurred during transcription: {e}"
            )
            self.log_display.appendPlainText(f"ERROR: {error_msg}")
            import traceback

            self.log_display.appendPlainText(traceback.format_exc())
            QMessageBox.critical(self, "Transcription Error", error_msg)
            self.status_bar.showMessage("Transcription failed.", 5000)
            self.text_display.setPlaceholderText("Transcription failed.")
        finally:
            self._update_ui_state()  # Always update UI state after transcription attempt

    @Slot()
    def _save_srt_file(self):
        """Opens a save file dialog and saves the current timed text data as an SRT."""
        if not self.timed_text_data:
            QMessageBox.warning(
                self,
                "No Data to Save",
                "There is no timed text data loaded to save as SRT.",
            )
            self._update_ui_state()  # Ensure button is disabled
            return

        if not MODEL_AVAILABLE or not hasattr(model, "save_timed_text_as_srt"):
            QMessageBox.critical(
                self,
                "Fatal Error",
                "Model module or save function not available. Cannot save SRT.",
            )
            self._update_ui_state()  # Ensure button is disabled
            return

        # Suggest filename based on original audio/SRT file if available
        if self.current_audio_path:
            default_filename = (
                Path(self.current_audio_path).with_suffix(".srt").name
            )
        elif self.file_path_lineedit.text():
            default_filename = (
                Path(self.file_path_lineedit.text()).with_suffix(".srt").name
            )
        else:
            default_filename = "transcript.srt"

        # Get last used directory from settings or use current file's dir or home
        last_dir = self.settings.value(
            "last_file_dir", str(Path.home()), type=str
        )
        if self.file_path_lineedit.text():
            last_dir = str(Path(self.file_path_lineedit.text()).parent)

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save SRT File",
            str(Path(last_dir) / default_filename),  # Suggest path + filename
            "SRT Files (*.srt);;All Files (*)",
        )

        if file_path:
            try:
                # Ensure file_path has .srt extension if not explicitly added by user
                if not file_path.lower().endswith(".srt"):
                    file_path += ".srt"

                self.log_display.appendPlainText(
                    f"Attempting to save SRT to: {Path(file_path).name}"
                )
                self.status_bar.showMessage(
                    f"Saving SRT to {Path(file_path).name}...", 0
                )
                QApplication.processEvents()  # Update GUI

                model.save_timed_text_as_srt(self.timed_text_data, file_path)

                self.settings.setValue(
                    "last_file_dir", str(Path(file_path).parent)
                )  # Save directory
                self.status_bar.showMessage(
                    f"SRT saved successfully to {Path(file_path).name}.", 5000
                )
                self.log_display.appendPlainText("SRT save successful.")

            except (OSError, IOError, model.SRTWriteError) as e:
                error_msg = f"Failed to save SRT file: {e}"
                self.log_display.appendPlainText(f"ERROR: {error_msg}")
                QMessageBox.critical(self, "Save Error", error_msg)
                self.status_bar.showMessage("SRT saving failed.", 5000)
            except Exception as e:
                error_msg = (
                    f"An unexpected error occurred during SRT save: {e}"
                )
                self.log_display.appendPlainText(f"ERROR: {error_msg}")
                import traceback

                self.log_display.appendPlainText(traceback.format_exc())
                QMessageBox.critical(self, "Save Error", error_msg)
                self.status_bar.showMessage("SRT saving failed.", 5000)
        else:
            self.status_bar.showMessage("SRT save cancelled.", 2000)

        self._update_ui_state()  # Ensure state is correct after save attempt

    def _display_timed_text(self):
        """Populates the text display with structured timed text and builds position maps."""
        if not self.timed_text_data:
            self.text_display.clear()
            self.text_display.setPlaceholderText(
                "No timed text data available."
            )
            self._clear_position_maps()
            self._previous_highlight_range = (0, 0)
            self._update_ui_state()  # Update controls
            return

        self.text_display.clear()
        # Build plain text string matching the expected layout (sentences separated by newlines)
        full_text_builder = []  # Use a list for efficient string building
        for i, sentence in enumerate(self.timed_text_data):
            sentence_text = sentence.get("text", "").strip()
            if sentence_text:  # Only add text if sentence is not empty
                full_text_builder.append(sentence_text)
            if i < len(self.timed_text_data) - 1:
                # Add newline AFTER a non-empty sentence, if it's not the last one
                if sentence_text:
                    full_text_builder.append("\n")

        full_text = "".join(full_text_builder)
        self.text_display.setPlainText(full_text)  # Set all text at once

        self._build_text_pos_maps()  # Build maps for all levels

        cursor = self.text_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        self.text_display.setTextCursor(cursor)
        self._clear_highlight()  # Clear any potential old highlight artifacts
        self._previous_highlight_range = (
            0,
            len(self.text_display.document().toPlainText()),
        )  # Reset tracking to cover full doc initially

        self.current_sim_index = 0
        # Highlight the first unit if data exists and maps are built
        if any(self._text_pos_maps.values()) and self._get_flat_timed_units():
            self._update_current_unit_highlight()  # This will set _previous_highlight_range

        self._update_ui_state()  # Update controls

    @Slot(str)
    def _on_highlight_level_changed(self, level: str):
        """Handles highlight level change."""
        self.log_display.appendPlainText(
            f"Highlight level changed to: {level}"
        )
        if self.timed_text_data:
            # Map is already built for all levels, just update highlight and reset index
            self._handle_stop()  # Stop any running simulation/playback (also clears highlight)
            self.current_sim_index = 0  # Reset index
            self._previous_highlight_range = (
                0,
                len(self.text_display.document().toPlainText()),
            )  # Reset tracking
            # Highlight the first unit at the new level IF there is data and a map for this level
            # The check self._text_pos_maps.get(level) ensures the map was successfully built for this level
            if self._text_pos_maps.get(level) and self._get_flat_timed_units():
                self._update_current_unit_highlight()  # This will set _previous_highlight_range
        self._update_ui_state()  # Update controls

    def _get_current_level_pos_map(self):
        """Returns the character position map for the currently selected highlight level."""
        level = self.highlight_level_combo.currentText()
        return self._text_pos_maps.get(
            level, {}
        )  # Return empty dict if level isn't valid

    def _get_flat_timed_units(self):
        """
        Returns a flat list of timed units (sentence, word, or char)
        from the structured data based on the selected level.
        Includes start_time, end_time, and source indices.
        This list is used for simulation iteration and mapping.
        """
        level = self.highlight_level_combo.currentText()
        flat_list = []
        if not self.timed_text_data:
            return flat_list

        # Generate a flat list of units with their source indices and times
        # Ensure time values are floats, default to 0.0 if missing/invalid
        # Use a unique index for the flat list for easier navigation
        flat_index_counter = 0

        for i, sentence in enumerate(self.timed_text_data):
            sentence_start = float(sentence.get("start_time", 0.0))
            sentence_end = float(
                sentence.get("end_time", sentence_start + 0.01)
            )  # Ensure end >= start
            if sentence_end < sentence_start:
                sentence_end = sentence_start + 0.01

            if level == "Sentence":
                if sentence.get(
                    "text", ""
                ).strip():  # Only add sentences with non-empty text
                    flat_list.append(
                        {
                            "start_time": sentence_start,
                            "end_time": sentence_end,
                            "source_sentence_index": i,
                            "source_word_index": -1,
                            "source_char_index": -1,
                            "flat_index": flat_index_counter,
                        }
                    )
                    flat_index_counter += 1
            else:  # Word or Character level
                words = sentence.get("words", [])
                # If sentence has text but no word data, handle as a single unit for Word level if selected
                if not words and sentence.get("text", "").strip():
                    if level == "Word":
                        flat_list.append(
                            {
                                "start_time": sentence_start,  # Use sentence time
                                "end_time": sentence_end,  # Use sentence time
                                "source_sentence_index": i,
                                "source_word_index": -1,  # Indicates sentence treated as word
                                "source_char_index": -1,
                                "flat_index": flat_index_counter,
                            }
                        )
                        flat_index_counter += 1
                    # Don't add sentence as a single char unit if no word data
                else:  # Process words (and characters within them) if words list is not empty
                    # Ensure word times are relative to the start of the audio, not segment start.
                    # The data loaded/transcribed should already have global times.
                    last_word_end_time = sentence_start  # Use sentence start as initial fallback
                    for j, word in enumerate(words):
                        word_text = word.get(
                            "text", ""
                        ).strip()  # Strip word text too
                        if not word_text:
                            continue  # Skip empty words

                        word_start = float(
                            word.get("start_time", last_word_end_time)
                        )
                        # Estimate end time if not provided, base it on start time and potential duration
                        # Fallback: If current word start <= previous word end, use previous end + small buffer
                        if j > 0 and word_start < last_word_end_time:
                            word_start = (
                                last_word_end_time + 0.001
                            )  # Ensure times are generally increasing

                        # Fallback for end time: estimate based on text length relative to sentence/segment duration
                        word_end = float(
                            word.get(
                                "end_time",
                                word_start + (len(word_text) * 0.05),
                            )
                        )  # Default minimal duration
                        if word_end < word_start:
                            word_end = word_start + (
                                len(word_text) * 0.05
                            )  # Ensure end >= start

                        if level == "Word":
                            flat_list.append(
                                {
                                    "start_time": word_start,
                                    "end_time": word_end,
                                    "source_sentence_index": i,
                                    "source_word_index": j,
                                    "source_char_index": -1,
                                    "flat_index": flat_index_counter,
                                }
                            )
                            flat_index_counter += 1
                        elif level == "Character":
                            char_duration = (
                                (word_end - word_start) / len(word_text)
                                if len(word_text) > 0
                                and (word_end - word_start) > 0
                                else 0.05
                            )
                            for k, char in enumerate(word_text):
                                # Character timing is an ESTIMATE based on word timing
                                char_start = word_start + (k * char_duration)
                                char_end = (
                                    char_start + char_duration
                                )  # Estimate end time
                                flat_list.append(
                                    {
                                        "start_time": char_start,
                                        "end_time": char_end,
                                        "source_sentence_index": i,
                                        "source_word_index": j,
                                        "source_char_index": k,
                                        "flat_index": flat_index_counter,
                                    }
                                )
                                flat_index_counter += 1
                        last_word_end_time = (
                            word_end  # Update for next word's fallback
                        )

        # Sort by start time (important for timer-based simulation)
        flat_list.sort(key=lambda x: x["start_time"])

        # Re-assign flat_index after sorting to ensure sequential numbering
        for i, unit in enumerate(flat_list):
            unit["flat_index"] = i

        return flat_list

    def _update_current_unit_highlight(self):
        """Highlights the text unit at the current simulation index using the position map."""
        if not self.timed_text_data or not any(self._text_pos_maps.values()):
            self._clear_highlight()
            self._previous_highlight_range = (
                0,
                len(self.text_display.document().toPlainText()),
            )
            return

        flat_units = self._get_flat_timed_units()
        current_level_map = self._get_current_level_pos_map()

        if (
            not flat_units
            or self.current_sim_index < 0
            or self.current_sim_index >= len(flat_units)
        ):
            self._clear_highlight()  # Ensure no highlight if index is invalid
            self._previous_highlight_range = (
                0,
                len(self.text_display.document().toPlainText()),
            )
            return

        # 1. Clear the previous highlight efficiently
        self._clear_highlight_range(*self._previous_highlight_range)

        # 2. Find the character range for the current unit using its source indices
        current_unit = flat_units[self.current_sim_index]

        # Determine the map key based on highlight level and unit's source indices
        level = self.highlight_level_combo.currentText()
        unit_key = None
        if level == "Sentence":
            # Ensure the unit corresponds to a Sentence level unit (e.g., source_word_index is -1)
            # This check is important if flat_units contains mixed types (though _get_flat_timed_units filters)
            if current_unit.get("source_word_index", 0) == -1:
                unit_key = (
                    current_unit.get("source_sentence_index", -1),
                    -1,
                    -1,
                )
        elif level == "Word":
            # Ensure the unit corresponds to a Word level unit
            if current_unit.get("source_char_index", 0) == -1:
                unit_key = (
                    current_unit.get("source_sentence_index", -1),
                    current_unit.get("source_word_index", -1),
                    -1,
                )
        elif level == "Character":
            # Ensure the unit corresponds to a Character level unit
            unit_key = (
                current_unit.get("source_sentence_index", -1),
                current_unit.get("source_word_index", -1),
                current_unit.get("source_char_index", -1),
            )

        start_char_pos, end_char_pos = (0, 0)  # Default invalid range

        if unit_key is not None and unit_key in current_level_map:
            start_char_pos, end_char_pos = current_level_map[unit_key]

            # 3. Apply highlighting to the new range
            cursor = self.text_display.textCursor()
            cursor.setPosition(start_char_pos)
            cursor.setPosition(end_char_pos, QTextCursor.MoveMode.KeepAnchor)
            cursor.mergeCharFormat(self.highlight_format)

            # 4. Scroll to make the highlighted area visible
            # Update the visible cursor to the start of the highlight so ensureCursorVisible works on it
            visible_cursor = self.text_display.textCursor()
            visible_cursor.setPosition(start_char_pos)
            self.text_display.setTextCursor(visible_cursor)
            self.text_display.ensureCursorVisible()

            # 5. Update the stored previous range for the next update cycle
            self._previous_highlight_range = (start_char_pos, end_char_pos)

        else:
            # This means the calculated flat unit index doesn't have a corresponding position in the map
            self.log_display.appendPlainText(
                f"WARNING: Could not find char position in map ({level}) for flat index {self.current_sim_index} (Key: {unit_key}). Map size: {len(current_level_map)}. Clearing highlight."
            )
            self._clear_highlight_range(
                start_char_pos, end_char_pos
            )  # Clear potential old highlight
            self._previous_highlight_range = (
                0,
                len(self.text_display.document().toPlainText()),
            )  # Reset range tracking to full doc

        # The status bar message is updated by the timer/playback slot that *calls* this method

    def _build_text_pos_maps(self):
        """
        Builds character position maps for Sentence, Word, and Character levels
        by iterating through the structured timed data and incrementally building
        the plain text content as it will appear in the QTextEdit.
        """
        self._clear_position_maps()  # Clear old maps
        if not self.timed_text_data:
            self.log_display.appendPlainText(
                "No timed text data to build position maps."
            )
            # Text display is cleared by _display_timed_text caller
            return

        self.log_display.appendPlainText("Building text position maps...")

        # Use the plain text already set in _display_timed_text as the source for mapping
        # This ensures consistency between the displayed text and the character positions
        full_text = self.text_display.document().toPlainText()
        if not full_text:
            self.log_display.appendPlainText(
                "WARNING: Text display is empty. Cannot build maps."
            )
            return  # Cannot build maps if there's no text

        current_char_pos = (
            0  # Tracks the current position in the full_text string
        )

        for s_idx, sentence_data in enumerate(self.timed_text_data):
            sentence_text_in_data = sentence_data.get("text", "").strip()
            if not sentence_text_in_data:
                # Skip map entry for empty sentences in data, but need to account for their
                # potential space/newlines if they were part of the original text build logic
                # The current mapping logic builds based on the *already set* plain text,
                # so we just need to find where this sentence's text starts in the full text.
                # This requires sequential scanning.

                # Alternative: Re-build the text AND the maps simultaneously
                # Let's revert to the simultaneous build approach, as it's more robust
                # against discrepancies between data structure and text display logic.
                pass  # Re-implementing the logic from _display_timed_text here

        # Simultaneous text build and map building (copied/adapted from _display_timed_text and previous _build_text_pos_maps)
        self.text_display.clear()  # Clear text *again* to rebuild precisely for mapping

        displayed_text_builder = []
        current_char_pos = 0

        for s_idx, sentence_data in enumerate(self.timed_text_data):
            sentence_text_in_data = sentence_data.get("text", "").strip()
            words = sentence_data.get("words", [])

            # Only process if there's content for this sentence (text or words)
            if not sentence_text_in_data and not words:
                continue  # Skip this sentence block entirely if it's empty in the data

            sentence_start_pos = current_char_pos  # Start position of the sentence block in the combined text

            # Build text from words and map word/char positions
            word_parts_text = []
            for w_idx, word_data in enumerate(words):
                word_text = word_data.get("text", "").strip()
                if not word_text:
                    continue  # Skip empty word entries

                if w_idx > 0:
                    displayed_text_builder.append(" ")
                    current_char_pos += 1

                word_start_pos = current_char_pos

                for k_idx, char_text in enumerate(word_text):
                    char_start_pos = current_char_pos
                    displayed_text_builder.append(char_text)
                    current_char_pos += 1
                    char_end_pos = current_char_pos
                    self._text_pos_maps["Character"][(s_idx, w_idx, k_idx)] = (
                        char_start_pos,
                        char_end_pos,
                    )

                word_end_pos = current_char_pos
                self._text_pos_maps["Word"][(s_idx, w_idx, -1)] = (
                    word_start_pos,
                    word_end_pos,
                )
                word_parts_text.append(word_text)

            # Handle case where sentence data has text but no words were parsed/added (e.g., SRT with only sentence timing)
            if not words and sentence_text_in_data:
                temp_word_text = sentence_text_in_data
                temp_word_start_pos = current_char_pos

                for k_idx, char_text in enumerate(temp_word_text):
                    char_start_pos = current_char_pos
                    displayed_text_builder.append(char_text)
                    current_char_pos += 1
                    char_end_pos = current_char_pos
                    self._text_pos_maps["Character"][(s_idx, -1, k_idx)] = (
                        char_start_pos,
                        char_end_pos,
                    )

                temp_word_end_pos = current_char_pos
                self._text_pos_maps["Word"][(s_idx, -1, -1)] = (
                    temp_word_start_pos,
                    temp_word_end_pos,
                )
                word_parts_text = [
                    temp_word_text
                ]  # Ensure sentence text is based on this block

            # Determine the final sentence text that was actually displayed
            # This should match the combined word parts, or the raw sentence text if no words
            actual_displayed_sentence_text = " ".join(word_parts_text).strip()
            if not actual_displayed_sentence_text and sentence_text_in_data:
                actual_displayed_sentence_text = (
                    sentence_text_in_data  # Fallback
                )

            # Store sentence position map IF the sentence resulted in some displayed text
            if actual_displayed_sentence_text:
                sentence_end_pos = current_char_pos
                self._text_pos_maps["Sentence"][(s_idx, -1, -1)] = (
                    sentence_start_pos,
                    sentence_end_pos,
                )

            # Add newline after the sentence block if it's not the last one and resulted in displayed text
            if (
                s_idx < len(self.timed_text_data) - 1
                and actual_displayed_sentence_text
            ):
                displayed_text_builder.append("\n")
                current_char_pos += 1  # Account for the newline

        # Set the final constructed text into the QTextEdit
        final_displayed_text = "".join(displayed_text_builder)
        self.text_display.setPlainText(final_displayed_text)

        total_mapped_units = sum(len(m) for m in self._text_pos_maps.values())
        self.log_display.appendPlainText(
            f"Built text position maps: {len(self._text_pos_maps['Sentence'])} sentences, {len(self._text_pos_maps['Word'])} words, {len(self._text_pos_maps['Character'])} chars. Total units mapped: {total_mapped_units}"
        )

        if total_mapped_units == 0 and self.timed_text_data:
            self.log_display.appendPlainText(
                "ERROR: Position maps are empty despite having timed text data. Highlighting will not work."
            )

    def _clear_highlight(self):
        """Removes highlighting from the entire text display."""
        # Use a cursor spanning the entire document
        cursor = self.text_display.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        # Apply the default format to remove highlight attributes
        cursor.mergeCharFormat(self.default_format)
        # Do NOT reset the visible cursor to the start, as this resets the scroll position.

    def _clear_highlight_range(self, start_pos: int, end_pos: int):
        """Removes highlighting from a specific text range."""
        # Check if the range is valid and within the document bounds
        doc_len = len(self.text_display.document().toPlainText())
        start_pos = max(0, min(start_pos, doc_len))
        end_pos = max(0, min(end_pos, doc_len))

        if start_pos >= end_pos:
            return  # Nothing to clear

        cursor = self.text_display.textCursor()
        cursor.setPosition(start_pos)
        cursor.setPosition(end_pos, QTextCursor.MoveMode.KeepAnchor)
        cursor.mergeCharFormat(self.default_format)

    @Slot()
    def _handle_play(self):
        """Starts or pauses audio playback and synchronized highlight update."""

        if not self.timed_text_data or not self.current_audio_path:
            QMessageBox.warning(
                self,
                "No Audio",
                "No timed text data loaded, or no audio file available for playback.",
            )
            return

        # Stop the simulation timer if it's running (Play Section mode)
        if self.simulation_timer.isActive():
            self.simulation_timer.stop()
            self.log_display.appendPlainText(
                "Stopped simulation timer (Play Section)."
            )

        # Ensure the audio source is set if not already or if it changed
        # Corrected: Use isEmpty() instead of isNull() for QUrl [3, 9]
        if (
            self.media_player.source().toLocalFile() != self.current_audio_path
            or self.media_player.source().isEmpty()
        ):
            self.media_player.setSource(
                QUrl.fromLocalFile(self.current_audio_path)
            )
            self.log_display.appendPlainText(
                f"Set audio source to: {Path(self.current_audio_path).name}"
            )
            # Setting source can be asynchronous. Seeking immediately might not work reliably.
            # For robustness, should ideally connect to mediaStatusChanged and seek when status is LoadedMedia.
            # For now, rely on Qt's internal handling which often works for local files.

        if (
            self.media_player.playbackState()
            == QMediaPlayer.PlaybackState.PlayingState
        ):
            self.media_player.pause()
            self.status_bar.showMessage("Playback paused.")
            self.log_display.appendPlainText("Paused audio playback.")
        elif (
            self.media_player.playbackState()
            == QMediaPlayer.PlaybackState.PausedState
        ):
            self.media_player.play()
            self.status_bar.showMessage("Playback resumed.")
            self.log_display.appendPlainText("Resumed audio playback.")
        else:  # Stopped or InvalidState
            self.log_display.appendPlainText(
                f"Starting audio playback: {Path(self.current_audio_path).name}"
            )

            # Determine start time based on current simulation index
            flat_units = self._get_flat_timed_units()
            start_time_sec = 0.0
            if flat_units and self.current_sim_index < len(flat_units):
                # Get the start time of the current unit, default to 0 if key missing
                start_time_sec = flat_units[self.current_sim_index].get(
                    "start_time", 0.0
                )
                # Set audio player position
                self.log_display.appendPlainText(
                    f"Attempting to seek audio to current unit time: {start_time_sec:.3f}s"
                )
                self.media_player.setPosition(int(start_time_sec * 1000))
            else:
                self.log_display.appendPlainText(
                    "No flat units or invalid index. Starting audio from beginning (0.000s)."
                )
                self.media_player.setPosition(0)

            self._ignore_playback_updates = False  # make sure we reset the flag before playing
            self.media_player.play()
            # Update highlight immediately and ensure scrollbar sync
            self._update_current_unit_highlight()
            self.status_bar.showMessage("Playing audio...")

        self._update_ui_state()  # Update controls

    @Slot()
    def _handle_play_section(self):
        """Starts or pauses text highlighting simulation (timer-based)."""
        if not self.timed_text_data or not any(self._text_pos_maps.values()):
            QMessageBox.warning(
                self,
                "No Data",
                "No timed text data or position maps available for simulation.",
            )
            return

        # Stop the media player if it's running (Play Audio mode)
        if (
            self.media_player.playbackState()
            != QMediaPlayer.PlaybackState.StoppedState
        ):
            self.media_player.stop()
            self.log_display.appendPlainText(
                "Stopped audio playback (Play Audio)."
            )

        flat_units = self._get_flat_timed_units()
        if not flat_units:
            self.log_display.appendPlainText("No units to simulate.")
            self.status_bar.showMessage("Simulation stopped (no units).", 3000)
            self._clear_highlight()  # Ensure no highlight
            self._previous_highlight_range = (
                0,
                len(self.text_display.document().toPlainText()),
            )
            self._update_ui_state()  # Update controls
            return

        if self.simulation_timer.isActive():
            self.simulation_timer.stop()
            self.status_bar.showMessage("Simulation paused.")
            self.log_display.appendPlainText("Paused simulation.")
        else:
            self.log_display.appendPlainText("Starting simulation...")

            # Start from the current simulation index (which might have been set by click/selection or navigation)
            if self.current_sim_index < 0 or self.current_sim_index >= len(
                flat_units
            ):
                self.current_sim_index = 0  # Reset if index is out of bounds

            # Get the start time of the current unit
            start_time_sec = flat_units[self.current_sim_index].get(
                "start_time", 0.0
            )
            self.log_display.appendPlainText(
                f"Starting simulation from unit index: {self.current_sim_index+1}/{len(flat_units)} (time: {start_time_sec:.3f}s)"
            )

            # Reset simulation state for timer-based run
            self.sim_start_time_real = time.time()
            self.sim_start_time_data = start_time_sec
            self._update_current_unit_highlight()  # Ensure the starting unit is highlighted immediately

            self.simulation_timer.start(
                DEFAULT_HIGHLIGHT_DURATION_MS
            )  # Update every N ms
            self.status_bar.showMessage("Simulation running...")

        self._update_ui_state()  # Update controls

    @Slot()
    def _handle_stop(self):
        """Stops both audio playback and text highlighting simulation, and clears highlight."""
        self._ignore_playback_updates = True  # make sure we reset the flag before stopping
        stopped_something = False
        if (
            self.media_player.playbackState()
            != QMediaPlayer.PlaybackState.StoppedState
        ):
            self.media_player.stop()
            self.status_bar.showMessage("Playback stopped.", 2000)
            self.log_display.appendPlainText("Audio playback stopped.")
            stopped_something = True

        if self.simulation_timer.isActive():
            self.simulation_timer.stop()
            if (
                not stopped_something
            ):  # Avoid redundant status if audio also stopped
                self.status_bar.showMessage("Simulation stopped.", 2000)
            self.log_display.appendPlainText("Simulation stopped.")
            stopped_something = True

        # Clear highlight after stopping, but only if something was actually stopped
        # or if there was active data that might have been highlighted.
        if stopped_something or (
            self.timed_text_data is not None and len(self.timed_text_data) > 0
        ):
            self._clear_highlight()
            # The _clear_highlight function already updates _previous_highlight_range to full doc.
            # self._previous_highlight_range = (0, len(self.text_display.document().toPlainText()))
            pass  # _clear_highlight handles the range update now.

        if not stopped_something:
            self.status_bar.showMessage(
                "Ready.", 1000
            )  # Default status if nothing was running

        self._update_ui_state()  # Update controls

    @Slot()
    def _update_highlight_from_timer(self):
        """Updates highlighted unit based on timer, for Play Section mode."""
        if not self.timed_text_data or not any(self._text_pos_maps.values()):
            self._handle_stop()  # Stop and clear highlight
            self.log_display.appendPlainText(
                "Simulation stopped: No data or map."
            )
            return  # Exit the update method

        flat_units = self._get_flat_timed_units()
        if not flat_units:
            self._handle_stop()  # Stop and clear highlight
            self.log_display.appendPlainText(
                "Simulation stopped: No flat units to simulate."
            )
            return

        # Calculate current simulation time based on timer and speed
        elapsed_real_time = time.time() - self.sim_start_time_real
        current_sim_time = self.sim_start_time_data + (
            elapsed_real_time * self.simulation_speed_multiplier
        )

        # Find the index of the unit that should be highlighted at the current time
        # Search forwards from the current index for efficiency
        next_index = self.current_sim_index  # Start search from current index

        # Optimization: If current time is *past* the current unit's end time, advance index
        # Search forwards until we find a unit whose end time is *after* or *equal to* the current time,
        # or we reach the end of the list.
        while next_index < len(
            flat_units
        ) - 1 and current_sim_time >= flat_units[next_index].get(
            "end_time", float("inf")
        ):
            next_index += 1

        # If current time is *before* the unit's start time, backtrack index.
        # This can happen if speed changes or simulation starts mid-unit.
        # Search backwards until we find a unit whose start time is *before* or *equal to* the current time,
        # or we reach the beginning of the list.
        while next_index > 0 and current_sim_time < flat_units[next_index].get(
            "start_time", 0.0
        ):
            next_index -= 1

        # Ensure index is within bounds (should already be due to loops, but safety)
        next_index = max(0, min(next_index, len(flat_units) - 1))

        # Handle end of simulation (check if time is past the very last unit)
        # Use the end time of the last unit for this check
        if flat_units and current_sim_time >= flat_units[-1].get(
            "end_time", float("inf")
        ):
            self._handle_stop()  # Stop and clear highlight
            self.status_bar.showMessage("Simulation finished.", 5000)
            self.log_display.appendPlainText(
                "Simulation finished (reached end of data)."
            )
            return  # Exit the update method

        # Update highlight if the current unit has changed
        if next_index != self.current_sim_index:
            self.current_sim_index = next_index
            self._update_current_unit_highlight()

        # Update status bar with simulation time
        current_unit_time = flat_units[self.current_sim_index].get(
            "start_time", 0.0
        )  # Show start time of current unit
        self.status_bar.showMessage(
            f"Sim Time: {current_sim_time:.3f}s / Unit: {self.current_sim_index + 1}/{len(flat_units)} ({current_unit_time:.3f}s)"
        )

    @Slot(int)
    def _update_highlight_from_playback(self, position_ms: int):
        """Updates highlighted unit based on media player position, for Play Audio mode."""
        if self._ignore_playback_updates:  # ignore updates if we are in play section mode
            return
        # Only update if the media player is actually playing or paused (e.g. seeking while paused)
        # Use positionChanged signal; this slot might be called even when stopped, but check state.
        if (
            self.media_player.playbackState()
            != QMediaPlayer.PlaybackState.StoppedState
            and self.timed_text_data
        ):
            current_audio_time = position_ms / 1000.0

            flat_units = self._get_flat_timed_units()
            if not flat_units or not any(self._text_pos_maps.values()):
                # Should not happen if playback is possible, but defensively clear
                self._clear_highlight()
                self._previous_highlight_range = (
                    0,
                    len(self.text_display.document().toPlainText()),
                )
                return  # Cannot highlight without data or maps

            # Find the index of the unit that should be highlighted at the current time
            # Search from current index for efficiency
            next_index = (
                self.current_sim_index
            )  # Start search from current index
            if next_index < 0 or next_index >= len(flat_units):
                next_index = 0  # Reset if out of bounds initially

            # Move forward if time has passed the current unit's end time
            while next_index < len(
                flat_units
            ) - 1 and current_audio_time >= flat_units[next_index].get(
                "end_time", float("inf")
            ):
                next_index += 1

            # Move backward if time is before the current unit's start time (e.g., seeking backward)
            while next_index > 0 and current_audio_time < flat_units[
                next_index
            ].get("start_time", 0.0):
                next_index -= 1

            # Ensure index is within bounds
            next_index = max(0, min(next_index, len(flat_units) - 1))

            # Handle end of playback (check if time is past the very last unit)
            if flat_units and current_audio_time >= flat_units[-1].get(
                "end_time", float("inf")
            ):
                # The playback state might still be Playing briefly at the very end.
                # Let the stateChanged signal handle the full stop logic.
                # Just ensure the last unit (or end) is highlighted briefly.
                if self.current_sim_index != len(flat_units) - 1:
                    self.current_sim_index = len(flat_units) - 1
                    self._update_current_unit_highlight()
                # Don't return here, let position continue updating until state becomes Stopped
                pass

            # Update highlight if the current unit has changed, but only if player is playing
            # If paused, we still want the highlight to update if the user *seeks*,
            # so update if index changed regardless of Paused/Playing.
            if next_index != self.current_sim_index:
                self.current_sim_index = next_index
                self._update_current_unit_highlight()

            # Update status bar with audio time
            # Only show time if player is not stopped
            if (
                self.media_player.playbackState()
                != QMediaPlayer.PlaybackState.StoppedState
            ):
                current_unit_time = flat_units[self.current_sim_index].get(
                    "start_time", 0.0
                )  # Show start time of current unit
                self.status_bar.showMessage(
                    f"Audio Time: {current_audio_time:.3f}s / Unit: {self.current_sim_index + 1}/{len(flat_units)} ({current_unit_time:.3f}s)"
                )

    @Slot(float)
    def _update_sim_speed(self, value: float):
        """Updates the simulation speed multiplier."""
        # Only apply if value is reasonable (spinbox already limits this, but defensive)
        if (
            0.0 < value <= 10.0
        ):  # Hardcoded cap, should match spinbox range roughly
            self.simulation_speed_multiplier = value
            self.status_bar.showMessage(
                f"Simulation speed set to {value:.1f}x", 2000
            )
            # If timer-based simulation is active, adjust timer base time to prevent jumps
            # Calculate the current elapsed simulation time based on the OLD speed
            if self.simulation_timer.isActive():
                # Need the old multiplier to calculate correctly.
                # Store the old value before updating self.simulation_speed_multiplier or retrieve from spinbox before it changes.
                # A simpler approach: just reset the base time calculation.
                self.sim_start_time_real = time.time()
                # The current_sim_time is already tracked by the timer update, no need to recalculate it here.
                # Just update the real time base so the *next* timer tick calculates correctly with the new speed.
                # The sim_start_time_data corresponds to the time *at* the sim_start_time_real reset.

        self._update_ui_state()  # Update controls (speed spinbox might affect playback button state logic, though maybe not strictly necessary here)

    @Slot()
    def _navigate_unit(self, direction: int):
        """Navigates to the previous or next timed unit."""
        if not self.timed_text_data or not any(self._text_pos_maps.values()):
            return

        flat_units = self._get_flat_timed_units()
        if not flat_units:
            return

        # Stop simulation/playback when manually navigating
        self._handle_stop()  # This also clears highlight and resets _previous_highlight_range

        # Calculate the new index
        new_index = self.current_sim_index + direction
        new_index = max(0, min(new_index, len(flat_units) - 1))

        # Only update if index changed
        if new_index != self.current_sim_index:
            self.current_sim_index = new_index
            # Re-apply highlight for the new current index AFTER stopping/clearing
            self._update_current_unit_highlight()

            # Update time base to the new position's start time for potential future Play/Play Section
            new_time_sec = flat_units[self.current_sim_index].get(
                "start_time", 0.0
            )
            self.sim_start_time_data = (
                new_time_sec  # For timer-based simulation
            )
            self.sim_start_time_real = (
                time.time()
            )  # Reset real time base for timer

            # If audio player is valid and has a source, seek it to the new position
            # Corrected: Use isEmpty() instead of isNull() for QUrl [3, 9]
            if (
                self.current_audio_path
                and not self.media_player.source().isEmpty()
            ):
                # Check if the player is in a state where setPosition is likely to work
                # LoadedMedia or PausedState or PlayingState might be ok
                # Avoid setting position if InvalidMedia or NoMedia
                if self.media_player.mediaStatus() not in [
                    QMediaPlayer.MediaStatus.NoMedia,
                    QMediaPlayer.MediaStatus.InvalidMedia,
                ]:
                    self.media_player.setPosition(int(new_time_sec * 1000))
                    self.log_display.appendPlainText(
                        f"Seeked audio to {new_time_sec:.3f}s"
                    )
                else:
                    self.log_display.appendPlainText(
                        f"Cannot seek audio, media player not ready (status: {self.media_player.mediaStatus()})."
                    )

            self.status_bar.showMessage(
                f"Navigated to unit {self.current_sim_index+1}/{len(flat_units)} ({new_time_sec:.3f}s)"
            )
        else:
            # Already at the first/last unit
            self.status_bar.showMessage(
                f"Already at {'first' if direction < 0 else 'last'} unit.",
                2000,
            )

        self._update_ui_state()  # Update controls

    @Slot()
    def _on_text_selection_changed(self):
        """
        Handles text selection changes. If the change indicates a cursor movement
        (click) or a selection, navigates to the unit under the cursor/selection start.
        """
        cursor = self.text_display.textCursor()

        # Get the character position at the start of the selection or the cursor position if no selection
        target_char_pos = (
            cursor.selectionStart()
            if cursor.hasSelection()
            else cursor.position()
        )

        # Only attempt to update index if data exists and maps are built
        if self.timed_text_data and any(self._text_pos_maps.values()):

            # Get the map for the current highlight level
            current_level_map = self._get_current_level_pos_map()
            flat_units = (
                self._get_flat_timed_units()
            )  # Need flat units to get flat_index

            if not flat_units or not current_level_map:
                # Maps or flat units are empty, cannot navigate by selection
                return

            # Find the unit whose character range contains the target position
            found_unit_index = -1
            level = self.highlight_level_combo.currentText()

            # Iterate through map keys and their ranges to find the containing unit.
            # Then find the flat index corresponding to that unit key.
            found_unit_key = None
            # Iterate through the map to find the key whose range contains the position
            for unit_key, (start_pos, end_pos) in current_level_map.items():
                # Check if the target position falls within this unit's character range
                # Use <= end_pos for robustness with clicks right at the end boundary of a character/word
                if target_char_pos >= start_pos and target_char_pos <= end_pos:
                    found_unit_key = unit_key
                    break  # Found the unit's key in the map

            if found_unit_key is not None:
                # Now find the flat index corresponding to this unit_key
                # Iterate through flat units to match the key
                for unit in flat_units:
                    unit_key_in_flat = (
                        unit.get("source_sentence_index", -1),
                        unit.get("source_word_index", -1),
                        unit.get("source_char_index", -1),
                    )
                    # Need to adjust the unit_key_in_flat based on the *level* being mapped
                    # to match how keys are stored in _text_pos_maps
                    if level == "Sentence":
                        flat_unit_lookup_key = (unit_key_in_flat[0], -1, -1)
                    elif level == "Word":
                        flat_unit_lookup_key = (
                            unit_key_in_flat[0],
                            unit_key_in_flat[1],
                            -1,
                        )
                    elif level == "Character":
                        # For Character level, the key in the map is the full (s,w,k) index
                        flat_unit_lookup_key = unit_key_in_flat
                    else:
                        continue  # Should not happen

                    if flat_unit_lookup_key == found_unit_key:
                        found_unit_index = unit.get("flat_index", -1)
                        break  # Found the corresponding flat unit index

            # Update current_sim_index and highlight if a unit was found and it's different
            if (
                found_unit_index != -1
                and found_unit_index != self.current_sim_index
            ):
                # Stop any ongoing playback/simulation FIRST
                self._handle_stop()  # This also clears the highlight and resets _previous_highlight_range

                self.current_sim_index = found_unit_index
                # Re-apply highlight for the new current index AFTER stopping/clearing
                self._update_current_unit_highlight()
                self.log_display.appendPlainText(
                    f"Navigated to unit {self.current_sim_index+1}/{len(flat_units)} by selection/cursor change."
                )

                # Update time base for Play/Play Section
                # Need to re-get flat units potentially if _handle_stop rebuilds them (it doesn't currently)
                flat_units_after_stop = self._get_flat_timed_units()
                if flat_units_after_stop and self.current_sim_index < len(
                    flat_units_after_stop
                ):
                    new_time_sec = flat_units_after_stop[
                        self.current_sim_index
                    ].get("start_time", 0.0)
                    self.sim_start_time_data = new_time_sec
                    self.sim_start_time_real = time.time()

                    # Seek audio player if active (even if paused/stopped, set position for next play)
                    # Corrected: Use isEmpty() instead of isNull() for QUrl [3, 9]
                    if (
                        self.current_audio_path
                        and not self.media_player.source().isEmpty()
                    ):
                        # Check if the player is in a state where setPosition is likely to work
                        # LoadedMedia or PausedState or PlayingState might be ok
                        # Avoid setting position if InvalidMedia or NoMedia
                        if self.media_player.mediaStatus() not in [
                            QMediaPlayer.MediaStatus.NoMedia,
                            QMediaPlayer.MediaStatus.InvalidMedia,
                            QMediaPlayer.MediaStatus.LoadingMedia,
                        ]:
                            self.media_player.setPosition(
                                int(new_time_sec * 1000)
                            )
                            self.log_display.appendPlainText(
                                f"Seeked audio to selection time: {new_time_sec:.3f}s"
                            )
                        else:
                            self.log_display.appendPlainText(
                                f"Cannot seek audio, media player not ready (status: {self.media_player.mediaStatus()})."
                            )

            # If unit found but index is the same, no action needed other than potentially updating time base
            # if the click was within the current unit (already handled above for the found_unit_index case)
            # If no unit found at click position - log warning or ignore? Ignoring is less intrusive.
            # The 'Could not find character position in map' warning in _update_current_unit_highlight is more relevant.

        # Update status bar to indicate selection exists (even if we didn't jump)
        elif cursor.hasSelection():
            self.status_bar.showMessage(
                "Text selected. 'Play Section' will start near selection.",
                2000,
            )
        else:
            self.status_bar.showMessage(
                "Ready.", 1000
            )  # Or similar default status

        self._update_ui_state()  # Update controls

    @Slot(QMediaPlayer.PlaybackState)
    def _media_player_state_changed(self, state: QMediaPlayer.PlaybackState):
        """Handles media player state changes."""
        # Note: Highlighting sync is primarily driven by positionChanged when Playing.
        # This slot is mainly for status bar and stopping timer if needed.
        if state == QMediaPlayer.PlaybackState.StoppedState:
            self.status_bar.showMessage("Playback stopped.", 2000)
            # The _handle_stop method (which might call this) already clears highlight and stops timer.
            # Ensure highlight is off and timer is stopped defensively
            if self.simulation_timer.isActive():
                self.simulation_timer.stop()
                self.log_display.appendPlainText(
                    "Simulation timer stopped by playback state change to Stopped."
                )
            self._clear_highlight()  # Ensure highlight is off - redundant if _handle_stop was called, but safe
            # The _clear_highlight function already updates _previous_highlight_range to full doc.
            # self._previous_highlight_range = (0, len(self.text_display.document().toPlainText()))

        elif state == QMediaPlayer.PlaybackState.PlayingState:
            self.status_bar.showMessage("Playing audio...", 0)
            # Ensure timer is stopped if audio is playing.
            if self.simulation_timer.isActive():
                self.simulation_timer.stop()
                self.log_display.appendPlainText(
                    "Simulation timer stopped by playback state change to Playing."
                )

        elif state == QMediaPlayer.PlaybackState.PausedState:
            self.status_bar.showMessage("Playback paused.", 2000)
            # Ensure timer is stopped if audio is paused.
            if self.simulation_timer.isActive():
                self.simulation_timer.stop()
                self.log_display.appendPlainText(
                    "Simulation timer stopped by playback state change to Paused."
                )

        self._update_ui_state()  # Update controls based on state

    def _show_settings_dialog(self):
        """Shows the settings dialog for style and theme."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Settings")
        layout = QFormLayout(dialog)

        # Style selection
        style_label = QLabel("Application Style:")
        style_combo = QComboBox()
        available_styles = QStyleFactory.keys()
        style_combo.addItems(available_styles)
        current_style_name = QApplication.instance().style().objectName()
        if current_style_name in available_styles:
            style_combo.setCurrentText(current_style_name)
        else:
            # Fallback to 'Fusion' or the first available style
            fallback_style = "Fusion"
            if fallback_style in available_styles:
                style_combo.setCurrentText(fallback_style)
            elif available_styles:
                style_combo.setCurrentIndex(0)
            else:
                style_combo.addItem(
                    "Default"
                )  # Add a dummy if no styles found (unlikely)
                style_combo.setCurrentText("Default")

        layout.addRow(style_label, style_combo)

        # Color Scheme selection
        theme_label = QLabel("Color Scheme:")
        theme_combo = QComboBox()
        theme_combo.addItems(["Auto", "Light", "Dark"])
        # Get current scheme from QApplication style hints if possible
        current_scheme_enum = (
            QApplication.instance().styleHints().colorScheme()
        )
        # Map enum to index
        current_scheme_index = 0  # Default to Auto
        if current_scheme_enum == Qt.ColorScheme.Light:
            current_scheme_index = 1
        elif current_scheme_enum == Qt.ColorScheme.Dark:
            current_scheme_index = 2

        # Use the currently active scheme for the dialog's initial state
        theme_combo.setCurrentIndex(current_scheme_index)

        layout.addRow(theme_label, theme_combo)

        # Font Size
        font_label = QLabel("Font Size:")
        font_spinbox = QSpinBox()
        font_spinbox.setRange(8, 24)
        # Get font size from text_display as it's the most visible text area
        font_spinbox.setValue(self.text_display.font().pointSize())
        layout.addRow(font_label, font_spinbox)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addRow(button_box)

        # Show dialog and apply settings if accepted
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_style = style_combo.currentText()
            selected_scheme_index = theme_combo.currentIndex()
            selected_font_size = font_spinbox.value()

            # Apply style and theme immediately
            app = QApplication.instance()  # Get the application instance
            if app:
                # Only set style if it's a valid, available style
                if selected_style in QStyleFactory.keys():
                    app.setStyle(QStyleFactory.create(selected_style))
                else:
                    self.log_display.appendPlainText(
                        f"WARNING: Selected style '{selected_style}' not available. Using default."
                    )

                scheme_enum = Qt.ColorScheme.Unknown  # 0: Auto
                if selected_scheme_index == 1:
                    scheme_enum = Qt.ColorScheme.Light
                elif selected_scheme_index == 2:
                    scheme_enum = Qt.ColorScheme.Dark
                app.styleHints().setColorScheme(scheme_enum)
                # Clear any potential QSS that might interfere with native style/palette
                app.setStyleSheet(
                    ""
                )  # Clear any previous stylesheet that might interfere

                # Apply font size
                # Setting font on the main window and explicit text areas is usually best
                font = self.font()
                font.setPointSize(selected_font_size)
                self.setFont(font)
                self.text_display.setFont(font)
                self.log_display.setFont(font)

                # Save settings
                self.settings.setValue("style", selected_style)
                self.settings.setValue("colorScheme", selected_scheme_index)
                self.settings.setValue("fontSize", selected_font_size)

                self.log_display.appendPlainText(
                    f"Settings applied: Style='{selected_style}', Scheme='{theme_combo.currentText()}', Font Size={selected_font_size}"
                )
            else:
                self.log_display.appendPlainText(
                    "ERROR: Cannot apply settings, QApplication instance not found."
                )

        self._update_ui_state()  # Ensure controls are updated based on any state changes

    def _apply_current_style_and_theme(self):
        """Applies the saved or default style and color scheme."""
        app = QApplication.instance()
        if not app:
            self.log_display.appendPlainText(
                "WARNING: QApplication instance not available to apply style/theme."
            )
            return

        style_name = self.settings.value("style", "Fusion", type=str)
        # Check if the saved style is still available
        available_styles = QStyleFactory.keys()
        if style_name not in available_styles:
            # Fallback if saved style isn't available
            fallback_style = "Fusion"
            if fallback_style in available_styles:
                style_name = fallback_style
                self.log_display.appendPlainText(
                    f"WARNING: Saved style '{self.settings.value('style', 'N/A')}' not available. Falling back to '{style_name}'."
                )
            elif available_styles:
                style_name = available_styles[0]  # Use first available
                self.log_display.appendPlainText(
                    f"WARNING: Saved style '{self.settings.value('style', 'N/A')}' not available. Falling back to first available style '{style_name}'."
                )
            else:
                self.log_display.appendPlainText(
                    "ERROR: No application styles available."
                )
                style_name = ""  # Indicate no style applied

        if style_name:
            app.setStyle(QStyleFactory.create(style_name))
            self.log_display.appendPlainText(f"Applied style: {style_name}")

        color_scheme_index = self.settings.value("colorScheme", 0, type=int)
        if not 0 <= color_scheme_index <= 2:
            color_scheme_index = 0

        # PySide6 6.9.0+ uses styleHints().setColorScheme
        scheme_enum = Qt.ColorScheme.Unknown  # 0: Auto
        if color_scheme_index == 1:
            scheme_enum = Qt.ColorScheme.Light
        elif color_scheme_index == 2:
            scheme_enum = Qt.ColorScheme.Dark

        app.styleHints().setColorScheme(scheme_enum)

        # Clear potential old stylesheet - QSS can override palette from style hints
        app.setStyleSheet("")

        self.log_display.appendPlainText(
            f"Applied color scheme: {['Auto', 'Light', 'Dark'][color_scheme_index]}"
        )

    def _load_settings(self):
        """Loads UI settings like window geometry, font size, style, theme."""
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        else:
            self.resize(DEFAULT_WINDOW_SIZE)

        # Font size loading is handled before applying style/theme so it affects widgets created later
        font_size = self.settings.value(
            "fontSize", DEFAULT_FONT_SIZE, type=int
        )
        font = self.font()  # Get window's default font
        font.setPointSize(font_size)
        self.setFont(font)  # Set window's font
        # QApplication.instance().setFont(font) # Avoid setting app-wide font here
        # Explicitly set for text areas to ensure consistency
        self.text_display.setFont(font)
        self.log_display.setFont(font)

        # Other settings are loaded/applied during _apply_current_style_and_theme

        self.log_display.appendPlainText("Settings loaded.")

    def closeEvent(self, event):
        """Handle window close event: Save settings and stop playback/simulation."""
        self._handle_stop()  # Stop any active processes
        self.settings.setValue("geometry", self.saveGeometry())
        # Save current style and color scheme from the QApplication instance
        app = QApplication.instance()
        if app:
            self.settings.setValue("style", app.style().objectName())
            # Corrected: Save integer enum value for color scheme
            # Using the index from the combo box is simpler for settings
            # Need to map current enum back to index
            current_scheme_enum = app.styleHints().colorScheme()
            scheme_index = 0  # Auto
            if current_scheme_enum == Qt.ColorScheme.Light:
                scheme_index = 1
            elif current_scheme_enum == Qt.ColorScheme.Dark:
                scheme_index = 2
            self.settings.setValue("colorScheme", scheme_index)
            # Font size is saved in the settings dialog or loaded on startup

        self.log_display.appendPlainText("Saving settings and exiting.")
        event.accept()


if __name__ == "__main__":
    # Check if running model.py tests
    # We do this check here in main.py's __main__ block
    # before QApplication is created or model is fully utilized by the GUI.
    if "--test" in sys.argv:
        # If --test is present, remove the PyQt arguments and run model.py tests.
        # This allows running 'python main.py --test X'
        # Need to strip PyQt args like -platform windows:darkmode=2
        # This isn't perfect but handles common cases.
        clean_args = [sys.argv[0]]  # Start with script name
        test_level = None
        i = 1
        while i < len(sys.argv):
            if sys.argv[i] == "--test" and i + 1 < len(sys.argv):
                # Found --test LEVEL
                try:
                    test_level = int(sys.argv[i + 1])
                    # Pass --test and level to model.py's argument parser later
                    clean_args.extend(sys.argv[i : i + 2])
                    i += 2
                except ValueError:
                    print(
                        "Error: --test requires a LEVEL argument when running main.py standalone.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
            elif sys.argv[i] == "--test":  # Found --test but no level
                print(
                    "Error: --test requires a LEVEL argument when running main.py standalone.",
                    file=sys.stderr,
                )
                sys.exit(1)
            elif not sys.argv[i].startswith(
                "-platform"
            ):  # Simple check to exclude platform args
                clean_args.append(sys.argv[i])
                i += 1
            else:
                # Skip this argument and check the next
                i += 1

        if test_level is None:
            # Should have exited above if --test was present without level
            pass  # This case should not be reached if --test was in sys.argv
        elif MODEL_AVAILABLE:
            print(
                f"Running model.py tests directly from main.py (__main__ block)...",
                file=sys.stderr,
            )
            # Call the test runner function from model.py
            # Pass the potentially cleaned arguments to model.run_model_tests
            # The model.py test runner needs to parse its own --test argument.
            # We only need to pass the test level value here, as run_model_tests takes int
            model.run_model_tests(test_level)
            # model.py's run_model_tests will sys.exit() after tests
            sys.exit(0)  # Should not be reached if model.py exits
        else:
            print(
                "ERROR: model.py not available. Cannot run tests.",
                file=sys.stderr,
            )
            sys.exit(1)

    # Otherwise, run the GUI application
    # Check for model.py availability before creating QApplication if possible,
    # but model import error is handled at the top.
    if not MODEL_AVAILABLE:
        # model.py print message already handled this
        sys.exit(1)

    # Check for PySide6 availability
    if not PYQT_AVAILABLE:
        # PySide6 print message already handled this
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setApplicationName(SETTINGS_APP)
    app.setOrganizationName(SETTINGS_ORG)
    app.setOrganizationDomain("ThereIsNoSource.com")

    window = MainWindow()

    window.show()

    sys.exit(app.exec())