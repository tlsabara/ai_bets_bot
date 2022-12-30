import os.path
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class InterfaceFridayDataManager(ABC):
    """
    Classe para apresentar ter a interface de derenciador de dados.
    """

    @abstractmethod
    def mount_dataframe(self, data:any) -> object:
        """
        Para criar os DF indendente da classe usada.
        :param data:
        :return:
        """
        ...

    @abstractmethod
    def export_to_file(self, df, **kwargs) -> None:
        """
        Para exportar os arquivos
        :param df:
        :param kwargs:
        :return:
        """
        ...


class FridayDataManagerPandas(InterfaceFridayDataManager):
    def mount_dataframe(self, data: any) -> object:
        return pd.DataFrame(data)

    def export_to_file(self, df: pd.DataFrame, **kwargs) -> None:
        dest_file = kwargs.get("dest_file")
        pth = Path(os.path.abspath(os.curdir)) / 'ocr_exports'
        pth.mkdir(exist_ok=True)
        dest_file = pth / dest_file
        df.to_csv(dest_file, sep="|")
