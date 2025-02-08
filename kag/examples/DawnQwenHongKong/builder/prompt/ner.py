# -*- coding: utf-8 -*-
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

import json
from string import Template
from typing import List
from kag.common.conf import KAG_PROJECT_CONF
from kag.interface import PromptABC
from knext.schema.client import SchemaClient


@PromptABC.register("dawn_person_ner")
class OpenIENERPrompt(PromptABC):
 
    template_zh = """
    {
        "instruction": "你是命名实体识别的专家。现在你需要从香港新闻媒体中提取与模式定义匹配的实体（以繁体字为主）。请注意！只需要提取和香港相关的实体，和香港无关的实体不必提取！如果不存在该类型的实体，请返回一个空列表。请以JSON字符串格式回应。你可以参照example进行抽取",
        "schema": $schema,
        "example": [
            {
                "input": "啟鑽商場酒樓冒煙消防救熄 疑抽煙系統失靈，鑽石山彩虹道啟鑽苑啟鑽商場一樓，今（8日）中午12時47分，一間中式酒樓廚房冒煙，職員報案求助，消防接報到場，開喉救熄，事件中無人受傷，毋須疏散。經調查後，相信是油炸食物時抽煙系統失靈，並無可疑。",
                "output": [
                            {
                                "name": "今（8日）中午12時47分",
                                "type": "Date",
                                "category": "Date",
                                "description": "香港啟鑽商場酒樓冒烟时间"
                            },
                            {
                                "name": "鑽石山彩虹道啟鑽苑啟鑽商場",
                                "type": "Location",
                                "category": "Location",
                                "description": "冒煙事故发生地点"
                            },
                            {
                                "name": "啟鑽商場酒樓冒煙消防救熄 疑抽煙系統失靈",
                                "type": "NewsArticale",
                                "category": "NewsArticale",
                                "description": "今（8日）中午12時47分 发生的消防新闻 "
                            },
                            {
                                "name": "消防",
                                "type": "Organization",
                                "category": "Organization",
                                "description": "香港消防队，负责该事故的调查"
                            },
                            {
                                "name": "抽煙系統",
                                "type": "Other",
                                "category": "Other",
                                "description": "一种排除烟雾系统，由于该系统失灵导致冒烟情况产生"
                            },
                            {
                                "name": "職員",
                                "type": "Person",
                                "category": "Person",
                                "description": "商场工作人员報案求助"
                            }
                        ]
            }
        ],
        "input": "$input"
    }    
        """
    template_en = template_zh
    def __init__(self, language: str = "", **kwargs):
        super().__init__(language, **kwargs)
        self.schema = SchemaClient(
            project_id=KAG_PROJECT_CONF.project_id
        ).extract_types()
        self.template = Template(self.template).safe_substitute(
            schema=json.dumps(self.schema)
        )

    @property
    def template_variables(self) -> List[str]:
        return ["input"]

    def parse_response(self, response: str, **kwargs):
        rsp = response
        if isinstance(rsp, str):
            rsp = json.loads(rsp)
        if isinstance(rsp, dict) and "output" in rsp:
            rsp = rsp["output"]
        if isinstance(rsp, dict) and "named_entities" in rsp:
            entities = rsp["named_entities"]
        else:
            entities = rsp

        return entities
