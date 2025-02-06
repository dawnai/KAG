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
        "instruction": "你是命名实体识别的专家。请从输入中提取与模式定义匹配的实体。如果不存在该类型的实体，请返回一个空列表。请尽可能严格schema中的实体内容进行抽取，保持知识图谱的准确性和精简性。并且以JSON字符串格式回应。你可以参照example进行抽取。",
        "schema": $schema,
        "example": [
            {
                "input": "特朗普宣布美国将再次退出《巴黎协定》，来源：新华网 | 2025年01月21日 09:12:47。原标题：特朗普宣布美国将再次退出《巴黎协定》新华社华盛顿1月20日电（记者刘亚南 邓仙来）美国总统特朗普20日签署行政令，宣布美国将再次退出旨在应对气候变化的《巴黎协定》。2015年，联合国气候变化大会达成《巴黎协定》，成为全球应对气候变化的重要成果。2017年6月，时任美国总统特朗普宣布美国将退出《巴黎协定》。2020年11月4日，美国正式退出该协定。此举遭到美国国内和国际社会的广泛批评。2021年1月20日，拜登就任总统首日签署行政令，宣布美国将重新加入《巴黎协定》。同年2月19日，美国正式重新加入《巴黎协定》。",
                "output": [
                        {"entity": "刘亚楠", "category": "Person"},
                        {"entity": "邓仙来", "category": "Person"},
                        {"entity": "特朗普", "category": "Person"},
                        {"entity": "联合国气候变化大会", "category": "Activity"},
                        {"entity": "拜登", "category": "Person"},
                        {"entity": "特朗普宣布美国将再次退出《巴黎协定》", "category": "NewsArticale"}
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
