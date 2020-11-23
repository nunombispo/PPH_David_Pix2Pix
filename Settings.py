import json


class Settings:
    def __init__(self):
        self.settingsFileName = "settings.json"
        self.settingsData = None

    def get_folder_source(self):
        return self.settingsData["folderSource"]

    def get_folder_output(self):
        return self.settingsData["folderOutput"]

    def get_folder_model(self):
        return self.settingsData["folderModel"]

    def get_epoch_number(self):
        return self.settingsData["epochNumber"]

    def get_original_image(self):
        return self.settingsData["originalFolder"]

    def get_original_model(self):
        return self.settingsData["originalModel"]

    def get_learning_rate(self):
        return self.settingsData["learningRate"]

    def get_batch_size(self):
        return self.settingsData["batchSize"]

    def set_folder_source(self, folder):
        self.settingsData["folderSource"] = folder

    def set_folder_output(self, folder):
        self.settingsData["folderOutput"] = folder

    def set_folder_model(self, folder):
        self.settingsData["folderModel"] = folder

    def set_epoch_number(self, number):
        self.settingsData["epochNumber"] = number

    def set_original_image(self, image):
        self.settingsData["originalFolder"] = image

    def set_original_model(self, model):
        self.settingsData["originalModel"] = model

    def set_learning_rate(self, learning_rate):
        self.settingsData["learningRate"] = learning_rate

    def set_batch_size(self, batch_size):
        self.settingsData["batchSize"] = batch_size


    def write_settings(self):
        try:
            with open(self.settingsFileName, 'w') as outfile:
                json.dump(self.settingsData, outfile, indent=4)
            return True
        except FileNotFoundError:
            print("Error saving Settings file")
            return False

    def read_settings(self):
        try:
            with open(self.settingsFileName) as json_file:
                self.settingsData = json.load(json_file)
                if self.settingsData is None:
                    self.create_settings()
        except FileNotFoundError:
            self.create_settings()

    def create_settings(self):
        self.settingsData = {
            "folderSource": "",
            "folderOutput": "",
            "folderModel": "",
            "learningRate": "",
            "batchSize": "",
            "epochNumber": "",
            "originalFolder": ""
        }

