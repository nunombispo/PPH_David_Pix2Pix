import subprocess
import shlex

from Settings import Settings
from PyQt5.QtWidgets import QTabWidget, QGroupBox, QFormLayout, QLabel, QLineEdit, QVBoxLayout, QDialogButtonBox, \
    QMessageBox, QTextEdit, QFileDialog, QPushButton
from AutoEncoder import AutoEncoder


class ExecuteTab(QTabWidget):
    def __init__(self):
        super().__init__()

        # Settings
        self.settings = Settings()
        self.settings.read_settings()

        self.line_edit_execute = None

        # Create Form
        self.formGroupBox = None
        self.create_form_groupbox()

        # Create layout
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.formGroupBox)
        self.setLayout(mainLayout)

    def create_form_groupbox(self):
        # Create form
        self.formGroupBox = QGroupBox("Execution Log")
        layout = QFormLayout()

        # Define QLineEdits
        self.line_edit_execute = QTextEdit()
        self.line_edit_execute.setReadOnly(True)

        # Buttons
        generate_model_button = QPushButton('Generate Model')
        generate_model_button.clicked.connect(self.training_model)
        run_model_button = QPushButton('Run Model')
        run_model_button.clicked.connect(self.run_model)

        # Add to layout
        layout.addRow(self.line_edit_execute)
        layout.addRow(generate_model_button)
        layout.addRow(run_model_button)
        self.formGroupBox.setLayout(layout)

    def read_settings(self):
        self.settings.read_settings()

    def training_model(self):
        self.line_edit_execute.setText("")
        auto_encoder = AutoEncoder(self.settings.get_folder_source(), self.settings.get_folder_output(),
                                  self.settings.get_folder_model(), self.line_edit_execute,
                                   self.settings.get_epoch_number(), self.settings.get_original_image(),
                                   self.settings.get_original_model())
        auto_encoder.run_thread(False)

    def run_model(self):
        self.line_edit_execute.setText("")
        auto_encoder = AutoEncoder(self.settings.get_folder_source(), self.settings.get_folder_output(),
                                  self.settings.get_folder_model(), self.line_edit_execute,
                                   self.settings.get_epoch_number(), self.settings.get_original_image(),
                                   self.settings.get_original_model())
        auto_encoder.run_thread(True)
