
import os
import logging
from kag.common.registry import import_modules_from_path

from kag.builder.runner import BuilderChainRunner

logger = logging.getLogger(__name__)


def buildKB(file_path):
    from kag.common.conf import KAG_CONFIG

    runner = BuilderChainRunner.from_config(KAG_CONFIG.all_config["kag_builder_pipeline"])
    runner.invoke(file_path)

    logger.info(f"\n\nbuildKB successfully for {file_path}\n\n")


if __name__ == "__main__":
    import_modules_from_path("./prompt")#记得使用自己的prompt模板
    dir_path = os.path.dirname(__file__)
    # 将file_path设置为之前准备好的语料文件路径
    file_path = os.path.join(dir_path, "data/trump.json")

    buildKB(file_path)