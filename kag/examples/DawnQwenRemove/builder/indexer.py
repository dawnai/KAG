# Copyright 2023 OpenSPG Authors
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied.
import os
import logging
from kag.common.registry import import_modules_from_path

from kag.builder.runner import BuilderChainRunner

logger = logging.getLogger(__name__)


def buildKB(file_path):
    from kag.common.conf import KAG_CONFIG #相关运行日志

    runner = BuilderChainRunner.from_config(KAG_CONFIG.all_config["kag_builder_pipeline"])
    runner.invoke(file_path)#invoke是核心函数，参数是接受待抽取数据

    logger.info(f"\n\nbuildKB successfully for {file_path}\n\n")


if __name__ == "__main__":
    import_modules_from_path("./prompt")#记得使用自己的prompt模板
    dir_path = os.path.dirname(__file__)
    # 将file_path设置为之前准备好的语料文件路径
    file_path = os.path.join(dir_path, "data/data.json")

    buildKB(file_path)
