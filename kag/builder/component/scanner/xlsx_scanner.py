from typing import Dict, List
import pandas as pd
from kag.interface import ScannerABC
from kag.common.utils import generate_hash_id
from knext.common.base.runnable import Input, Output
import os
import re
@ScannerABC.register("xlsx")
@ScannerABC.register("xlsx_scanner")
class XLSXScanner(ScannerABC):
    def __init__(
        self,
        header: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__(rank=rank, world_size=world_size)
        self.header = header

    @property
    def input_types(self) -> Input:
        return str

    @property
    def output_types(self) -> Output:
        return Dict

    def clean_text(self,text: str) -> str:
        """
        清洗文本，去除无关字符和多余空格。
        """
        # 替换中文空格（\u3000）和其他特殊空白符为普通空格
        text = re.sub(r'\s+', ' ', text)
        # 去除多余的换行符和制表符
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\t+', ' ', text)
        text = re.sub(r'\u200d', '', text)
        # 去除开头和结尾的多余空格
        text = text.strip()
        return text
    def clean_data(self,data: List[Dict]) -> List[Dict]:
        """
        清洗数据列表中的 title 和 content 字段。
        """
        cleaned_data = []
        for item in data:
            cleaned_item = {
                "id": item["id"],  # 保留原始 ID
                "title": self.clean_text(item["title"]),  # 清洗 title
                "content": self.clean_text(item["content"])  # 清洗 content
            }
            cleaned_data.append(cleaned_item)
        return cleaned_data
    def load_data(self, input: Input, **kwargs) -> List[Output]:
        """
        Loads data from an XLSX file and converts it into a list of dictionaries.
        Only 'title' and 'content' columns are processed.

        Args:
            input (Input): The input file path to the XLSX file.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Output]: A list of dictionaries containing the processed data.
        """
        input = self.download_data(input)  # 预处理输入路径
        if self.header:
            data = pd.read_excel(input, dtype=str)
        else:
            data = pd.read_excel(input, dtype=str, header=None)
            # 如果没有表头，假设第一列是 title，第二列是 content，希望刘博的新闻媒体是有表头的
            data.columns = ["title", "content"]

        # 确保只处理 'title' 和 'content' 列
        required_columns = ["title", "content"]
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Required columns {required_columns} not found in the file.")

        contents = []
        for _, row in data.iterrows():
            title = row["title"]
            content = row["content"]
            contents.append(
                {
                    "id": generate_hash_id(content),  # 使用 content 生成哈希 ID
                    "title": title,
                    "content": content
                }
            )
        # 清洗数据
        cleaned_data = self.clean_data(contents)

        print("扫描结果为：",cleaned_data)
        return cleaned_data
    


    # 数据预处理，使用默认的就行
    # def download_data(self, input: Input, **kwargs) -> str:
    #     """
    #     Downloads data from a given input URL or returns the input directly if it is not a URL.

    #     Args:
    #         input (Input): The input source, which can be a URL (starting with "http://" or "https://") or a local path.
    #         **kwargs: Additional keyword arguments (currently unused).

    #     Returns:
    #         str: The local file path if the input is a URL, or the input itself if it is not a URL.
    #     """
    #     if input.startswith("http://") or input.startswith("https://"):
    #         from kag.common.utils import download_from_http

    #         local_file_path = os.path.join(KAG_PROJECT_CONF.ckpt_dir, "file_scanner")
    #         if not os.path.exists(local_file_path):
    #             os.makedirs(local_file_path)
    #         local_file = os.path.join(local_file_path, os.path.basename(input))
    #         local_file = download_from_http(input, local_file)
    #         return local_file
    #     return input