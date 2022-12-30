import os
from abc import ABC
from pathlib import Path


class Configs(ABC):
    ROOT_DIR = Path('C:\\Users\\root\\Documents\\Git\\bot_spacerman')

    def __new__(cls, *args, **kwargs):
        if cls.__name__ == 'Configs':
            raise NotImplementedError("A classe não pode ser instanciada")


class WindowsConfigs(Configs):
    ROOT_TESSERACT = Path('C:\\Program Files\\Tesseract-OCR')
    TESSERACT_EXE = ROOT_TESSERACT / 'tesseract.exe'

class LunuxConfigs(Configs):
    ROOT_TESSERACT = Path.home()
    TESSERACT_EXE = ROOT_TESSERACT / 'tesseract'

