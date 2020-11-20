from pathlib import Path

from Settings import Settings
from PyQt5.QtWidgets import QTabWidget, QGroupBox, QFormLayout, QLabel, QLineEdit, QVBoxLayout, QDialogButtonBox, \
    QMessageBox, QTextEdit, QFileDialog, QPushButton


class SettingsTab(QTabWidget):
    def __init__(self):
        super().__init__()

        # Settings
        self.settings = Settings()

        # Define QLineEdits
        self.lineedit_folder_source = None
        self.lineedit_folder_output = None


        # Create Form
        self.formGroupBox = None
        self.create_form_groupbox()

        # Create buttons
        buttonBox = QDialogButtonBox(QDialogButtonBox.Save)
        buttonBox.accepted.connect(self.save_settings)

        # Create layout
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.formGroupBox)
        mainLayout.addWidget(buttonBox)
        self.setLayout(mainLayout)

    def create_form_groupbox(self):
        # Create form
        self.formGroupBox = QGroupBox("Folder Settings")
        layout = QFormLayout()

        # Create line edits
        self.lineedit_folder_source = QLineEdit()
        self.lineedit_folder_output = QLineEdit()
        self.lineedit_folder_model = QLineEdit()
        self.lineedit_epoch_number = QLineEdit()
        self.lineedit_original_image = QLineEdit()
        self.lineedit_original_model = QLineEdit()

        # Create buttons
        folder_source_button = QPushButton('Browse')
        folder_source_button.clicked.connect(self.folder_source_button_clicked)
        folder_output_button = QPushButton('Browse')
        folder_output_button.clicked.connect(self.folder_output_button_clicked)
        folder_model_button = QPushButton('Browse')
        folder_model_button.clicked.connect(self.folder_model_button_clicked)
        original_image_button = QPushButton('Browse')
        original_image_button.clicked.connect(self.original_image_button_clicked)
        original_model_button = QPushButton('Browse')
        original_model_button.clicked.connect(self.original_model_button_clicked)

        # Add to layout
        layout.addRow(QLabel())
        layout.addRow(QLabel("Folder Source:"))
        layout.addRow(self.lineedit_folder_source)
        layout.addRow(folder_source_button)
        layout.addRow(QLabel())
        layout.addRow(QLabel("Folder Output:"))
        layout.addRow(self.lineedit_folder_output)
        layout.addRow(folder_output_button)
        layout.addRow(QLabel())
        layout.addRow(QLabel("Folder Model:"))
        layout.addRow(self.lineedit_folder_model)
        layout.addRow(folder_model_button)
        layout.addRow(QLabel())
        layout.addRow(QLabel("Epoch Number:"))
        layout.addRow(self.lineedit_epoch_number)
        layout.addRow(QLabel())
        layout.addRow(QLabel())
        layout.addRow(QLabel())
        layout.addRow(QLabel("Run Images Folder:"))
        layout.addRow(self.lineedit_original_image)
        layout.addRow(original_image_button)
        layout.addRow(QLabel())
        layout.addRow(QLabel("Run Model file:"))
        layout.addRow(self.lineedit_original_model)
        layout.addRow(original_model_button)
        self.formGroupBox.setLayout(layout)

    def original_model_button_clicked(self):
        home = str(Path.home())
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename = QFileDialog.getExistingDirectory(self, caption='Pick folder', directory=home, options=options)
        if filename:
            self.lineedit_original_model.setText(filename)
        else:
            self.lineedit_original_model.setText('')

    def original_image_button_clicked(self):
        home = str(Path.home())
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename = QFileDialog.getExistingDirectory( self, caption='Pick folder', directory=home, options=options)
        if filename:
            self.lineedit_original_image.setText(filename)
        else:
            self.lineedit_original_image.setText('')

    def folder_source_button_clicked(self):
        home = str(Path.home())
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename = QFileDialog.getExistingDirectory(self, caption='Pick folder', directory=home, options=options)
        if filename:
            self.lineedit_folder_source.setText(filename)
        else:
            self.lineedit_folder_source.setText('')

    def folder_output_button_clicked(self):
        home = str(Path.home())
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename = QFileDialog.getExistingDirectory(self, caption='Pick folder', directory=home, options=options)
        if filename:
            self.lineedit_folder_output.setText(filename)
        else:
            self.lineedit_folder_output.setText('')

    def folder_model_button_clicked(self):
        home = str(Path.home())
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename = QFileDialog.getExistingDirectory(self, caption='Pick folder', directory=home, options=options)
        if filename:
            self.lineedit_folder_model.setText(filename)
        else:
            self.lineedit_folder_model.setText('')

    def read_settings(self):
        self.settings.read_settings()
        self.lineedit_folder_source.setText(self.settings.get_folder_source())
        self.lineedit_folder_output.setText(self.settings.get_folder_output())
        self.lineedit_folder_model.setText(self.settings.get_folder_model())
        self.lineedit_epoch_number.setText(str(self.settings.get_epoch_number()))
        self.lineedit_original_image.setText(str(self.settings.get_original_image()))
        self.lineedit_original_model.setText(str(self.settings.get_original_model()))

    def save_settings(self):
        self.settings.set_folder_source(self.lineedit_folder_source.text())
        self.settings.set_folder_output(self.lineedit_folder_output.text())
        self.settings.set_folder_model(self.lineedit_folder_model.text())
        self.settings.set_epoch_number(self.lineedit_epoch_number.text())
        self.settings.set_original_image(self.lineedit_original_image.text())
        self.settings.set_original_model(self.lineedit_original_model.text())
        success = self.settings.write_settings()

        # Check for success
        if success:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Settings saved successfully")
            msg.setWindowTitle("Information")
            msg.exec_()
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("There was an error saving the settings")
            msg.setWindowTitle("Error")
            msg.exec_()