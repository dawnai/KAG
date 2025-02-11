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
from typing import List

from kag.interface import PromptABC


@PromptABC.register("dawn_person_triple")
class OpenIETriplePrompt(PromptABC):
    template_zh = """
    {
        "instruction": "您是一位专门从事开放信息提取（OpenIE）的专家。请从input字段的文本中提取可能的关系（包括主语、谓语、宾语），并按照JSON格式列出它们，须遵循example字段的示例格式。请注意以下要求：1. 每个三元组应至少包含entity_list实体列表中的一个，但最好是两个命名实体。2. 明确地将代词解析为特定名称，以保持清晰度。",
        "entity_list": $entity_list,
        "input": "$input",
        "example": {
            "input": "深圳南山今日（8日）發生巴士撞上巴士站，事後釀成2死1傷。深圳公安通報事件。通報指出，2月8日10時許，在南山區沙河西路茶光村公交站，一公交車入站停靠時因司機突發疾病與站台發生碰撞，造成3名候車乘客受傷，其中2人經搶救無效死亡。經對公交車司機進行呼氣式酒精測試，結果為0mg/100ml。目前，事故正在進一步調查處理中。",
            "entity_list": [
                {"name": "2月8日10時", "category": "Date"},
                {"name": "深圳南山區沙河西路茶光村公交站", "category": "Location"},
                {"name": "深圳南山發生巴士撞上巴士站，事後釀成2死1傷", "category": "NewsArticale"},
                {"name": "深圳公安", "category": "Organization"},
                {"name": "呼氣式酒精測試", "category": "Other"},
            ],
            "output":[
                ["深圳南山區沙河西路茶光村公交站", "发生", "巴士撞上巴士站事故"],
                ["深圳公安", "通报", "造成3名候車乘客受傷，其中2人經搶救無效死亡"],
                ["公交车司机", "进行", "呼氣式酒精測試"],
                ["呼氣式酒精測試", "结果为", "0mg/100ml"],
                ["2月8日10時", "发生", "深圳南山發生巴士撞上巴士站，事後釀成2死1傷"]
            ]
        }
    }    
        """

    template_en = template_zh
    @property
    def template_variables(self) -> List[str]:
        return ["entity_list", "input"]

    def parse_response(self, response: str, **kwargs):
        rsp = response
        if isinstance(rsp, str):
            rsp = json.loads(rsp)
        if isinstance(rsp, dict) and "output" in rsp:
            rsp = rsp["output"]
        if isinstance(rsp, dict) and "triples" in rsp:
            triples = rsp["triples"]
        else:
            triples = rsp

        standardized_triples = []
        for triple in triples:
            if isinstance(triple, list):
                standardized_triples.append(triple)
            elif isinstance(triple, dict):
                s = triple.get("subject")
                p = triple.get("predicate")
                o = triple.get("object")
                if s and p and o:
                    standardized_triples.append([s, p, o])

        return standardized_triples
