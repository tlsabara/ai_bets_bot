from datetime import datetime
from pathlib import Path

from PIL import Image
import numpy as np
import pytesseract
import cv2
from project_globals import WindowsConfigs
from interfaces import InterfaceFridayDataManager
import os


class FridayDataGenerator:
    def __init__(self, data_manager: InterfaceFridayDataManager):
        if isinstance(data_manager, InterfaceFridayDataManager):
            self.data_mananger = data_manager
        self.utils = FridayUtils()
        self.workfolder = WindowsConfigs.ROOT_DIR / 'img_dst'
        self.collected_data = []
        self.files_indexed = []
        self._workfolder = WindowsConfigs.ROOT_DIR / 'img_dst'
        self.df = None

    def set_subdir_as_workdir(self, folder_name):
        """
        aplica o nome do subdiretorio como o diretorio de trabalho.
        :param folder_name:
        :return:
        """
        self.workfolder = self.workfolder / folder_name
        self._workfolder = self.workfolder / folder_name

    def get_fileindex_on_folder(self, sub_dir=None, flag=False):
        """
        Monta o indice de arquivos a serem passados no ocr
        :param sub_dir:
        :param flag:
        :return:
        """
        self._workfolder = self._workfolder / sub_dir if sub_dir else self._workfolder
        file_list = os.listdir(self._workfolder)
        for i in file_list:
            context = self._workfolder / i
            if context.is_file():
                self.files_indexed.append(context)
            else:
                temp = self._workfolder
                self.get_fileindex_on_folder(sub_dir=i)
                self._workfolder = temp

    def collect_data_from_indexed_files(self, verbose=False):
        """
        Pega a lista de arquivos indexados e rea;
        :param verbose:
        :return:
        """
        pytesseract.pytesseract.tesseract_cmd = WindowsConfigs.ROOT_TESSERACT / 'tesseract.exe'
        for i in self.files_indexed:
            data_dict = self.utils.get_data_from_image_name(i.name)
            ocr_text = self.utils.get_string_from_image(str(i))
            ocr_clean_text = self.utils.text_cleaner(ocr_text)
            crash_val = self.utils.transform_collected_text(ocr_clean_text)
            ocr_text = ocr_text.strip().replace('\n', '').replace('"', '').replace("'", '')
            data_dict['text'] = ocr_text
            data_dict['crash_value'] = crash_val
            self.collected_data.append(data_dict)
            if verbose:
                print('Dados:')
                for k, v in data_dict.items():
                    print(k, v, sep=': ')
                print('txt: ', ocr_clean_text, 'crsh: ', crash_val)
                print('--')

    def save_collected_data(self, dest_file: str = None) -> None:
        """
        Para salvar os dados coletados no arquivo especificado, via interface.
        :param dest_file:
        :return:
        """
        dest_file = dest_file if dest_file else f'export__{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        self.df = self.data_mananger.mount_dataframe(self.collected_data)
        self.data_mananger.export_to_file(self.df, dest_file=dest_file)


class FridayUtils:
    """
    Complemento da Friday
    """

    def __init__(self):
        self.ocr_worker = Path(os.path.abspath(os.curdir)) / 'ocr_tmp'
        self.ocr_worker.mkdir(exist_ok=True)

    def text_cleaner(self, text):
        """
        Realiza a limpeza do texto coletado.
        :param text:
        :return:
        """
        possible_values = 'I X . K'.split(' ')
        text = text.upper().strip().replace(' ', '') \
            .replace('I', '').replace('L', '').replace('K', '') \
            .replace('X', '').replace('‘', '').replace('(', '').replace(')', '') \
            .replace('\n', '').replace('"', '').replace("'", '')
        if len(text) > 1:
            if text[-1] in possible_values:
                text = text[0:-1]
        if len(text) == 0:
            print('len 0')
            text = -1
        return text

    def transform_collected_text(self, text, flag=False):
        """
        Transforma o texto str em float.
        :param text:
        :param flag:
        :return:
        """
        try:
            result = float(text)
        except ValueError or TypeError:
            print(f'Não convertido: {text}')
            result = -1
        except Exception as e:
            print("Erro não tratado: ", e)
            result = -2
        return result

    def get_data_from_image_name(self, filename: str) -> dict:
        """
        Retira os dados do nome do arquivo. Seguindo o padrão que utilizamos
        :param filename:
        :return:
        """
        splited = self.__check_dots(filename)
        extension = splited[1]
        splited = self.__check_underlines(splited[0])
        return {
            'filename': filename,
            'type': splited[0],
            'extension': extension,
            'year': splited[2],
            'month': splited[3],
            'day': splited[4],
            'hour': splited[5],
            'minutes': splited[6],
            'seconds': splited[7],
            'text': None,
            'crash_value': None
        }

    @staticmethod
    def __check_dots(splited):
        """
        [SOLID] Faz a separação da extensão e do nome do arquivo
        :param splited:
        :return:
        """
        splited = splited.split('.')
        if len(splited) != 2:
            raise ValueError('O nome do arquivo não tem o formato aceito.')
        return splited

    @staticmethod
    def __check_underlines(splited):
        """
        [SOLID] Faz a separação dos nomes do arquivo, seguindo o modelo tipo_save_ano_mes_dia_h_min_seg
        :param splited:
        :return:
        """
        splited = splited.split('_')
        if len(splited) != 8:
            raise ValueError('O tem mais valores que o permitido.')
        return splited

    def get_string_from_image(self, img_path: str) -> str:
        """
        Usando o tesseract para ler as imagens e coletar os crashses.
        Le o arquivo e apaga
        :param img_path:
        :return:
        """
        img_filename = Path(img_path).name
        img_filename = img_filename.replace('.', '_')
        # Read image with opencv
        img = cv2.imread(img_path)
        # Convert to gray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply dilation and erosion to remove some noise
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        # Write the image after apply opencv to do some ...
        thres_file = str(self.ocr_worker / f"{img_filename}_ocr@{datetime.now().strftime('%Y%m%d_%H%M%S')}__thres.png")
        cv2.imwrite(thres_file, img)
        # Recognize text with tesseract for python
        result = pytesseract.image_to_string(Image.open(thres_file))
        Path(thres_file).unlink()
        return result


if __name__ == '__main__':
    from interfaces import FridayDataManagerPandas

    f = FridayDataGenerator(FridayDataManagerPandas())
    s_dir = 'coleta_20221224_00_05'
    f.get_fileindex_on_folder(sub_dir=s_dir)
    f.collect_data_from_indexed_files(verbose=True)
    f.save_collected_data()