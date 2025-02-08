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
        "instruction": "你是命名实体识别的专家。现在你需要从香港新闻媒体中提取与模式定义匹配的实体（以繁体字为主）。如果不存在该类型的实体，请返回一个空列表,请注意！只需要提取和香港相关的实体，和香港无关的实体不必提取。请以JSON字符串格式回应。你可以参照example进行抽取",
        "schema": $schema,
        "example": [
            {
                "input": "深圳南山今日（8日）發生巴士撞上巴士站，事後釀成2死1傷。深圳公安通報事件。通報指出，2月8日10時許，在南山區沙河西路茶光村公交站，一公交車入站停靠時因司機突發疾病與站台發生碰撞，造成3名候車乘客受傷，其中2人經搶救無效死亡。經對公交車司機進行呼氣式酒精測試，結果為0mg/100ml。目前，事故正在進一步調查處理中。",
                "output": [
                            {
                                "name": "2月8日10時",
                                "type": "Date",
                                "category": "Date",
                                "description": "深圳南山發生巴士撞上巴士站时间"
                            },
                            {
                                "name": "深圳南山區沙河西路茶光村公交站",
                                "type": "Location",
                                "category": "Location",
                                "description": "事故发生地点"
                            },
                            {
                                "name": "深圳南山發生巴士撞上巴士站，事後釀成2死1傷",
                                "type": "NewsArticale",
                                "category": "NewsArticale",
                                "description": "2月8日10時 发生的事故新闻 "
                            },
                            {
                                "name": "深圳公安",
                                "type": "Organization",
                                "category": "Organization",
                                "description": "深圳市公安局，负责该事故的调查"
                            },
                            {
                                "name": "呼氣式酒精測試",
                                "type": "Other",
                                "category": "Other",
                                "description": "呼氣式酒精測試，一种测试驾驶员是否酒驾的测试工具，该事故的驾驶员测试結果為0mg/100ml，没有酒驾"
                            },
                            {
                                "name": "公交車司機",
                                "type": "Person",
                                "category": "Person",
                                "description": "该事故的公交车驾驶员，因为突發疾病與站台發生碰撞"
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
