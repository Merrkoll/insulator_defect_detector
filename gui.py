import os
import sys
import subprocess
import csv
from datetime import datetime
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QTextEdit, QComboBox, QFrame, QSizePolicy,
    QGridLayout, QMessageBox
)

from predict import predict_image


MODEL_OPTIONS = {
    'ResNet18': ('resnet18', 'insulator_model_resnet18.pth'),
    'EfficientNet-B0': ('efficientnet_b0', 'insulator_model_efficientnet_b0.pth'),
    'MobileNetV2': ('mobilenet_v2', 'insulator_model_mobilenet_v2.pth'),
}


class TrainWorker(QThread):
    finished_ok = pyqtSignal(str)
    finished_error = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    def run(self):
        try:
            self.log.emit(f'Запуск обучения модели {self.model_name}...')
            process = subprocess.run(
                ['python', 'train_model.py', '--model_name', self.model_name],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )

            if process.stdout:
                self.log.emit(process.stdout.strip())

            if process.returncode == 0:
                self.finished_ok.emit(f'Обучение модели {self.model_name} завершено успешно.')
            else:
                err_text = process.stderr.strip() if process.stderr else 'Неизвестная ошибка.'
                self.finished_error.emit(err_text)

        except Exception as e:
            self.finished_error.emit(str(e))


class ImageDropLabel(QLabel):
    file_dropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setObjectName('imagePreview')
        self.setText('Перетащите изображение сюда\nили нажмите "Загрузить файл"')
        self.setMinimumHeight(380)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                path = urls[0].toLocalFile().lower()
                if path.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.file_dropped.emit(path)
            event.acceptProposedAction()


class InsulatorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_file = None
        self.original_pixmap = None
        self.train_worker = None

        self.reports_dir = os.path.join(os.getcwd(), 'reports')
        os.makedirs(self.reports_dir, exist_ok=True)

        self.setWindowTitle('Детектор дефектов изоляторов ЛЭП')
        self.setMinimumSize(1200, 860)

        self._apply_styles()
        self._build_ui()
        self.statusBar().showMessage('Готово к работе')

    def open_reports_folder(self):
        try:
            os.makedirs(self.reports_dir, exist_ok=True)
            os.startfile(self.reports_dir)
            self.append_log(f'Открыта папка отчётов: {self.reports_dir}')
        except Exception as e:
            self.append_log(f'Не удалось открыть папку отчётов: {e}')
            QMessageBox.warning(
                self,
                'Ошибка',
                f'Не удалось открыть папку отчётов:\n{e}'
            )

    def predict_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self,
            'Выберите папку с изображениями'
        )

        if not folder_path:
            return

        model_name, model_file = self.get_selected_model()

        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(valid_ext) and os.path.isfile(os.path.join(folder_path, f))
        ]

        if not files:
            self.append_log('В выбранной папке нет подходящих изображений.')
            return

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_path = os.path.join(
            self.reports_dir,
            f'classification_log_{model_name}_{timestamp}.csv'
        )
        rows = []

        self.append_log(f'Запуск пакетной классификации: {folder_path}')
        self.append_log(f'Найдено изображений: {len(files)}')

        for file_name in files:
            file_path = os.path.join(folder_path, file_name)

            try:
                label, conf = predict_image(
                    model_file,
                    file_path,
                    model_name=model_name
                )

                rows.append({
                    'имя файла': file_name,
                    'архитектура модели': model_name,
                    'результат': label,
                    'уверенность модели': f'{conf:.4f}'
                })

                self.append_log(
                    f'[{model_name}] {file_name} -> {label} ({conf:.2%})'
                )

            except Exception as e:
                rows.append({
                    'имя файла': file_name,
                    'архитектура модели': model_name,
                    'результат': f'Ошибка: {e}',
                    'уверенность модели': ''
                })
                self.append_log(f'Ошибка для {file_name}: {e}')

        try:
            with open(log_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        'имя файла',
                        'архитектура модели',
                        'результат',
                        'уверенность модели'
                    ]
                )
                writer.writeheader()
                writer.writerows(rows)

            self.append_log(f'Лог сохранён: {log_path}')
            QMessageBox.information(
                self,
                'Готово',
                f'Пакетная классификация завершена.\nЛог сохранён:\n{log_path}'
            )

        except Exception as e:
            self.append_log(f'Не удалось сохранить лог: {e}')
            QMessageBox.warning(
                self,
                'Ошибка',
                f'Не удалось сохранить лог:\n{e}'
            )

    def _update_preview_pixmap(self):
        if self.original_pixmap is None:
            return

        target_size = self.image_label.size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return

        scaled = self.original_pixmap.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_preview_pixmap()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(18)

        # Левая панель
        self.sidebar = QFrame()
        self.sidebar.setObjectName('sidebar')
        self.sidebar.setMinimumWidth(260)
        self.sidebar.setMaximumWidth(320)
        self.sidebar.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(20, 20, 20, 20)
        sidebar_layout.setSpacing(16)

        app_title = QLabel('Insulator\nDetector')
        app_title.setObjectName('appTitle')

        app_subtitle = QLabel('Классификация состояния изоляторов по изображению')
        app_subtitle.setObjectName('appSubtitle')
        app_subtitle.setWordWrap(True)

        model_label = QLabel('Архитектура модели')
        model_label.setObjectName('sectionLabel')

        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_OPTIONS.keys())

        self.model_info = QLabel('Выберите модель для инференса или обучения.')
        self.model_info.setObjectName('hintLabel')
        self.model_info.setWordWrap(True)

        self.btn_load = QPushButton('Загрузить файл')
        self.btn_load.clicked.connect(self.load_file)

        self.btn_predict = QPushButton('Классифицировать')
        self.btn_predict.setObjectName('primaryButton')
        self.btn_predict.clicked.connect(self.predict)

        self.btn_predict_folder = QPushButton('Классифицировать папку')
        self.btn_predict.setObjectName('primaryButton')
        self.btn_predict_folder.clicked.connect(self.predict_folder)

        self.btn_reports = QPushButton('Отчёты')
        self.btn_reports.clicked.connect(self.open_reports_folder)

        self.result_card = QFrame()
        self.result_card.setObjectName('resultCard')
        result_layout = QVBoxLayout(self.result_card)
        result_layout.setContentsMargins(16, 16, 16, 16)
        result_layout.setSpacing(8)

        result_title = QLabel('Результат')
        result_title.setObjectName('cardTitle')

        self.result_badge = QLabel('Нет данных')
        self.result_badge.setObjectName('neutralBadge')
        self.result_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.result_conf = QLabel('Уверенность: —')
        self.result_conf.setObjectName('metricLabel')

        self.result_model = QLabel('Модель: —')
        self.result_model.setObjectName('metricLabel')

        result_layout.addWidget(result_title)
        result_layout.addWidget(self.result_badge)
        result_layout.addWidget(self.result_conf)
        result_layout.addWidget(self.result_model)

        sidebar_layout.addWidget(app_title)
        sidebar_layout.addWidget(app_subtitle)
        sidebar_layout.addSpacing(6)
        sidebar_layout.addWidget(model_label)
        sidebar_layout.addWidget(self.model_combo)
        sidebar_layout.addWidget(self.model_info)
        sidebar_layout.addSpacing(8)
        sidebar_layout.addWidget(self.btn_load)
        sidebar_layout.addWidget(self.btn_predict)
        sidebar_layout.addWidget(self.btn_predict_folder)
        sidebar_layout.addWidget(self.btn_reports)
        sidebar_layout.addSpacing(10)
        sidebar_layout.addWidget(self.result_card)
        sidebar_layout.addStretch()

        # Правая часть
        content = QVBoxLayout()
        content.setSpacing(18)

        header = QFrame()
        header.setObjectName('panel')
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(18, 16, 18, 16)

        header_title_box = QVBoxLayout()
        title = QLabel('Система автоматического выявления дефектов')
        title.setObjectName('pageTitle')

        subtitle = QLabel('Загрузите изображение, выберите обученную архитектуру и выполните классификацию.')
        subtitle.setObjectName('pageSubtitle')
        subtitle.setWordWrap(True)

        header_title_box.addWidget(title)
        header_title_box.addWidget(subtitle)

        header_layout.addLayout(header_title_box)
        header_layout.addStretch()

        preview_panel = QFrame()
        preview_panel.setObjectName('panel')
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(18, 18, 18, 18)
        preview_layout.setSpacing(12)

        preview_title = QLabel('Предпросмотр изображения')
        preview_title.setObjectName('cardTitle')

        self.image_label = ImageDropLabel()
        self.image_label.file_dropped.connect(self.set_image)
        self.image_label.setFixedHeight(300)

        self.file_name_label = QLabel('Файл не выбран')
        self.file_name_label.setObjectName('hintLabel')
        self.file_name_label.setWordWrap(True)
        self.file_name_label.setMaximumHeight(40)

        preview_layout.addWidget(preview_title)
        preview_layout.addWidget(self.image_label)
        preview_layout.addWidget(self.file_name_label)

        bottom_grid = QGridLayout()
        bottom_grid.setSpacing(18)

        info_panel = QFrame()
        info_panel.setObjectName('panel')
        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(18, 18, 18, 18)
        info_layout.setSpacing(8)

        info_title = QLabel('Информация')
        info_title.setObjectName('cardTitle')

        self.info_text = QLabel(
            'Поддерживаются модели ResNet18, EfficientNet-B0 и MobileNetV2.\n'
            'Для анализа используются сохранённые веса .pth из корня проекта.'
        )
        self.info_text.setObjectName('infoText')
        self.info_text.setWordWrap(True)

        info_layout.addWidget(info_title)
        info_layout.addWidget(self.info_text)
        info_layout.addStretch()

        log_panel = QFrame()
        log_panel.setObjectName('panel')
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(18, 18, 18, 18)
        log_layout.setSpacing(8)

        log_title = QLabel('Журнал')
        log_title.setObjectName('cardTitle')

        self.result = QTextEdit()
        self.result.setReadOnly(True)
        self.result.setPlaceholderText('Здесь будут отображаться действия программы...')

        log_layout.addWidget(log_title)
        log_layout.addWidget(self.result)

        bottom_grid.addWidget(info_panel, 0, 0)
        bottom_grid.addWidget(log_panel, 0, 1)
        bottom_grid.setColumnStretch(0, 1)
        bottom_grid.setColumnStretch(1, 2)

        content.addWidget(header)
        content.addWidget(preview_panel)
        content.addLayout(bottom_grid)

        root_layout.addWidget(self.sidebar)
        root_layout.addLayout(content, stretch=1)

    def _apply_styles(self):
        self.setStyleSheet("""
            QLabel#appTitle,
            QLabel#appSubtitle,
            QLabel#hintLabel,
            QLabel#sectionLabel,
            QLabel#pageTitle,
            QLabel#pageSubtitle,
            QLabel#cardTitle,
            QLabel#metricLabel,
            QLabel#infoText {
                background: transparent;
            }
            QWidget {
                background-color: #0f1117;
                color: #e8ecf1;
                font-family: 'Segoe UI';
                font-size: 14px;
            }

            QMainWindow {
                background-color: #0f1117;
            }

            QFrame#sidebar, QFrame#panel, QFrame#resultCard {
                background-color: #171a22;
                border: 1px solid #252a36;
                border-radius: 18px;
            }

            QFrame#panel {
                background-color: #151923;
            }

            QLabel#appTitle {
                font-size: 28px;
                font-weight: 700;
                color: #ffffff;
            }

            QLabel#appSubtitle, QLabel#hintLabel {
                color: #97a3b6;
                font-size: 13px;
            }

            QLabel#sectionLabel {
                font-size: 13px;
                font-weight: 600;
                color: #b8c2d1;
                margin-top: 4px;
            }

            QLabel#pageTitle {
                font-size: 26px;
                font-weight: 700;
                color: #ffffff;
            }

            QLabel#pageSubtitle, QLabel#infoText {
                color: #a5afbf;
                font-size: 14px;
            }

            QLabel#cardTitle {
                font-size: 16px;
                font-weight: 600;
                color: #ffffff;
            }

            QLabel#metricLabel {
                color: #c8d2df;
                font-size: 14px;
            }

            QLabel#imagePreview {
                background-color: #11151d;
                border: 2px dashed #2d8cff;
                border-radius: 16px;
                color: #8fa2bd;
                font-size: 16px;
                font-weight: 500;
                padding: 24px;
            }

            QComboBox, QTextEdit {
                background-color: #10141c;
                border: 1px solid #283041;
                border-radius: 12px;
                padding: 10px 12px;
                color: #eef3f8;
            }

            QComboBox:hover, QTextEdit:hover {
                border: 1px solid #3a4660;
            }

            QComboBox::drop-down {
                border: none;
                width: 28px;
            }

            QPushButton {
                background-color: #1d2430;
                border: 1px solid #2d3545;
                border-radius: 12px;
                padding: 12px 14px;
                font-size: 14px;
                font-weight: 600;
                color: #eef3f8;
            }

            QPushButton:hover {
                background-color: #273142;
                border: 1px solid #41506b;
            }

            QPushButton:pressed {
                background-color: #111722;
            }

            QPushButton#primaryButton {
                background-color: #2d8cff;
                border: 1px solid #2d8cff;
                color: white;
            }

            QPushButton#primaryButton:hover {
                background-color: #1f7ef2;
                border: 1px solid #1f7ef2;
            }

            QTextEdit {
                selection-background-color: #2d8cff;
            }

            QStatusBar {
                background-color: #0c0f15;
                color: #93a0b4;
            }
        """)

    def get_selected_model(self):
        ui_name = self.model_combo.currentText()
        return MODEL_OPTIONS[ui_name]

    def append_log(self, text: str):
        self.result.append(text)
        self.statusBar().showMessage(text, 5000)

    def set_image(self, file_path: str):
        if not file_path or not os.path.exists(file_path):
            self.append_log('Выбранный файл не существует.')
            return

        self.current_file = file_path
        pixmap = QPixmap(file_path)

        if pixmap.isNull():
            self.append_log('Не удалось загрузить изображение.')
            return

        self.original_pixmap = pixmap
        self._update_preview_pixmap()

        self.file_name_label.setText(f'Файл: {os.path.basename(file_path)}')
        self.append_log(f'Файл загружен: {file_path}')

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            'Выберите изображение',
            '',
            'Images (*.jpg *.jpeg *.png *.bmp *.webp)'
        )
        if file_path:
            self.set_image(file_path)

    def train_model(self):
        model_name, model_file = self.get_selected_model()

        reply = QMessageBox.question(
            self,
            'Подтверждение',
            f'Запустить обучение модели {model_name}?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        self.btn_train.setEnabled(False)
        self.btn_predict.setEnabled(False)
        self.btn_load.setEnabled(False)

        self.train_worker = TrainWorker(model_name)
        self.train_worker.log.connect(self.append_log)
        self.train_worker.finished_ok.connect(
            lambda msg: self._on_train_success(msg, model_file)
        )
        self.train_worker.finished_error.connect(self._on_train_error)
        self.train_worker.start()

        self.append_log(f'Обучение запущено: {model_name}')

    def _on_train_success(self, msg: str, model_file: str):
        self.append_log(msg)
        self.append_log(f'Модель сохранена или обновлена: {model_file}')
        self.btn_train.setEnabled(True)
        self.btn_predict.setEnabled(True)
        self.btn_load.setEnabled(True)

    def _on_train_error(self, err: str):
        self.append_log(f'Ошибка обучения: {err}')
        self.btn_train.setEnabled(True)
        self.btn_predict.setEnabled(True)
        self.btn_load.setEnabled(True)

    def predict(self):
        if not self.current_file:
            self.append_log('Сначала загрузите изображение.')
            return

        model_name, model_file = self.get_selected_model()

        try:
            label, conf = predict_image(
                model_file,
                self.current_file,
                model_name=model_name
            )

            self.result_badge.setText(label)
            self.result_conf.setText(f'Уверенность: {conf:.2%}')
            self.result_model.setText(f'Модель: {model_name}')

            if label == 'Исправный':
                self.result_badge.setStyleSheet("""
                    QLabel {
                        background-color: #173326;
                        color: #79e2a4;
                        border: 1px solid #29593f;
                        border-radius: 12px;
                        padding: 10px;
                        font-size: 18px;
                        font-weight: 700;
                    }
                """)
            else:
                self.result_badge.setStyleSheet("""
                    QLabel {
                        background-color: #3a1d25;
                        color: #ff9fb1;
                        border: 1px solid #6a3040;
                        border-radius: 12px;
                        padding: 10px;
                        font-size: 18px;
                        font-weight: 700;
                    }
                """)

            self.append_log(f'[{model_name}] Класс: {label} (доверие: {conf:.2%})')

        except FileNotFoundError:
            self.append_log(
                f'Файл модели не найден: {model_file}. Сначала обучите выбранную модель.'
            )
        except Exception as e:
            self.append_log(f'Ошибка классификации: {e}')


def main():
    app = QApplication(sys.argv)
    win = InsulatorGUI()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()