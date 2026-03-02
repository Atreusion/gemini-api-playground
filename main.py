import os
import sys

from dotenv import load_dotenv
import google.generativeai as genai

from PySide6.QtCore import QThread, Signal
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


class GeminiWorker(QThread):
    """Runs generate_content() in a background thread to keep the GUI responsive."""

    result_ready = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, model: genai.GenerativeModel, prompt: str) -> None:
        super().__init__()
        self._model = model
        self._prompt = prompt

    def run(self) -> None:
        try:
            response = self._model.generate_content(self._prompt)
            self.result_ready.emit(response.text)
        except Exception as exc:  # noqa: BLE001
            self.error_occurred.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self, model: genai.GenerativeModel) -> None:
        super().__init__()
        self._model = model
        self._worker: GeminiWorker | None = None

        self.setWindowTitle("Gemini API Playground")
        self.resize(800, 600)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        layout.addWidget(QLabel("Prompt:"))
        self._input = QTextEdit()
        self._input.setFixedHeight(100)
        self._input.setPlaceholderText("Enter your prompt here…")
        layout.addWidget(self._input)

        self._submit_btn = QPushButton("Submit")
        self._submit_btn.clicked.connect(self._on_submit)
        layout.addWidget(self._submit_btn)

        layout.addWidget(QLabel("Response:"))
        self._output = QTextEdit()
        self._output.setReadOnly(True)
        self._output.setPlaceholderText("Response will appear here…")
        layout.addWidget(self._output)

    def _on_submit(self) -> None:
        prompt = self._input.toPlainText().strip()
        if not prompt:
            return

        if self._worker is not None and self._worker.isRunning():
            return

        self._submit_btn.setEnabled(False)
        self._output.setPlainText("Generating…")

        self._worker = GeminiWorker(self._model, prompt)
        self._worker.result_ready.connect(self._on_result)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.finished.connect(lambda: self._submit_btn.setEnabled(True))
        self._worker.start()

    def _on_result(self, text: str) -> None:
        self._output.setPlainText(text)

    def _on_error(self, message: str) -> None:
        self._output.setPlainText("")
        QMessageBox.critical(self, "Error", message)


def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print(
            "Error: GEMINI_API_KEY is not set. "
            "Create a .env file based on .env.example.",
            file=sys.stderr,
        )
        sys.exit(1)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    app = QApplication(sys.argv)
    window = MainWindow(model)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
