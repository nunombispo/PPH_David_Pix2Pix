import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout, QTabWidget
from SettingsTab import SettingsTab
from ExecuteTab import ExecuteTab


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pix2Pix UI")
        self.resize(650, 900)
        self.table_widget = MyTableWidget(self)
        self.setCentralWidget(self.table_widget)
        self.show()

    def closeEvent(self, event):
        pass


class MyTableWidget(QWidget):

    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.layout = QVBoxLayout(self)

        # Initialize tab screen
        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self.tab_changed)
        self.tab1 = ExecuteTab()
        self.tab2 = SettingsTab()
        self.tabs.resize(300, 200)

        # Add tabs
        self.tabs.addTab(self.tab1, "Execute")
        self.tabs.addTab(self.tab2, "Encoder Settings")

        # Add tabs to widget
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    def tab_changed(self, index):
        if index == 0:
            self.tab1.read_settings()
        if index == 1:
            self.tab2.read_settings()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
