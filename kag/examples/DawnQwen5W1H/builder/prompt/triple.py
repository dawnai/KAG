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
        "instruction": "您是一位专门从事开放信息提取（OpenIE）的专家。请从input字段的文本中提取可能的关系（包括主语、谓语、宾语），并按照JSON格式列出它们，须遵循example字段的示例格式。请注意以下要求：1. 每个三元组应至少包含entity_list实体列表中的一个，但最好是两个命名实体。2. 明确地将代词解析为特定名称，以保持清晰度。3.三元组内容和香港相关",
        "entity_list": $entity_list,
        "input": "$input",
        "example": {
            "input": "啟鑽商場酒樓冒煙消防救熄 疑抽煙系統失靈，鑽石山彩虹道啟鑽苑啟鑽商場一樓，今（8日）中午12時47分，一間中式酒樓廚房冒煙，職員報案求助，消防接報到場，開喉救熄，事件中無人受傷，毋須疏散。經調查後，相信是油炸食物時抽煙系統失靈，並無可疑。",
            "entity_list": [
                {"name": "今（8日）中午12時47分", "category": "Date"},
                {"name": "鑽石山彩虹道啟鑽苑啟鑽商場", "category": "Location"},
                {"name": "啟鑽商場酒樓冒煙消防救熄 疑抽煙系統失靈", "category": "JiuLongEvent"},
                {"name": "消防", "category": "Organization"},
                {"name": "抽煙系統", "category": "Other"},
                {"name": "職員", "category": "Person"},
            ],
            "output":[
                ["鑽石山彩虹道啟鑽苑啟鑽商場", "发生", "冒烟情况"],
                ["消防", "调查", "油炸食物時抽煙系統失靈，並無可疑"],
                ["消防", "接報", "開喉救熄"],
                ["職員", "報案", "求助"],
                ["今（8日）中午12時47分", "发生", "啟鑽商場酒樓冒煙消防救熄 疑抽煙系統失靈"]
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
