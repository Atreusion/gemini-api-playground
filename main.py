"""Gemini API Playground GUI application using PySide6."""

import os
import sys

from dotenv import load_dotenv
from google import genai
from google.genai import chats
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

load_dotenv()


class PromptTextEdit(QTextEdit):
    """Custom QTextEdit that submits on Enter and inserts newline on Shift+Enter."""

    submit_requested = Signal()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        """Override keyPressEvent to handle Enter and Shift+Enter."""
        # Check if Enter/Return was pressed without Shift
        if (
            event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
            and not event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self.submit_requested.emit()
            event.accept()
        else:
            # For Shift+Enter or any other key, use default behavior
            super().keyPressEvent(event)


class GeminiWorker(QThread):
    """Runs generate_content() in a background thread to keep the GUI responsive."""

    result_ready = Signal(str)
    error_occurred = Signal(str)
    token_count_updated = Signal(int)

    def __init__(self, chat: chats.Chat, prompt: str) -> None:
        """Initialize the worker with the chat object and prompt."""
        super().__init__()
        self._chat = chat
        self._prompt = prompt

    def run(self) -> None:
        """Run the chat request in a separate thread."""
        try:
            response = self._chat.send_message(self._prompt)
            if response.usage_metadata:
                self.token_count_updated.emit(response.usage_metadata.total_token_count)
            self.result_ready.emit(response.text)
        except Exception as e:  # noqa: BLE001
            self.error_occurred.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window for the Gemini API Playground."""

    def __init__(self, chat: chats.Chat) -> None:
        """Initialize the main window with the chat object."""
        super().__init__()
        self._chat = chat

        self.setWindowTitle("Gemini API Playground")
        self.resize(800, 600)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        layout.addWidget(QLabel("Prompt:"))
        self._input = PromptTextEdit()
        self._input.setFixedHeight(100)
        self._input.setPlaceholderText(
            "Enter your prompt here (Enter to submit, Shift+Enter for new line)...",
        )
        self._input.submit_requested.connect(self._on_submit)
        layout.addWidget(self._input)

        self._submit_btn = QPushButton("Submit")
        self._submit_btn.clicked.connect(self._on_submit)
        layout.addWidget(self._submit_btn)

        layout.addWidget(QLabel("Chat History:"))
        self._output = QTextEdit()
        self._output.setReadOnly(True)
        self._output.setPlaceholderText("Chat history will appear here...")
        layout.addWidget(self._output)

        self._token_label = QLabel("Total tokens used: 0")
        layout.addWidget(self._token_label)

    def _on_submit(self) -> None:
        """Handle the submit action: get the prompt, disable the button, start the worker thread."""
        prompt = self._input.toPlainText().strip()
        if not prompt:
            return

        self._submit_btn.setEnabled(False)

        # Append user prompt to output
        if self._output.toPlainText():
            self._output.append("\n" + "=" * 80 + "\n")
        self._output.append(f"User: {prompt}\n")
        self._output.append("AI: Generating...")

        # Clear input field
        self._input.clear()

        self._worker = GeminiWorker(self._chat, prompt)
        self._worker.result_ready.connect(self._on_result_ready)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.token_count_updated.connect(self._on_token_count_updated)
        self._worker.finished.connect(lambda: self._submit_btn.setEnabled(True))
        self._worker.start()

    def _on_token_count_updated(self, token_count: int) -> None:
        """Update the token count label with the new total."""
        current_tokens = int(self._token_label.text().split(": ")[1])
        total_tokens = current_tokens + token_count
        self._token_label.setText(f"Total tokens used: {total_tokens}")

    def _on_result_ready(self, text: str) -> None:
        """Handle the result from the worker thread."""
        # Remove the "Generating..." placeholder
        current_text = self._output.toPlainText()
        if current_text.endswith("AI: Generating..."):
            current_text = current_text[: -len("Generating...")]
            self._output.setPlainText(current_text + text)
        else:
            self._output.append(text)

    def _on_error(self, message: str) -> None:
        """Handle errors from the worker thread."""
        self._output.setPlainText("")
        QMessageBox.critical(self, "Error", message)
        self._submit_btn.setEnabled(True)


def main() -> None:
    """Run the application."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(  # noqa: T201
            "Error: GEMINI_API_KEY is not set. Create a .env file based on .env.example.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = genai.Client()
    chat = client.chats.create(model="gemini-3-flash-preview")
    app = QApplication(sys.argv)
    window = MainWindow(chat)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
