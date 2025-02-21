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


@PromptABC.register("dawn_person_std")
class OpenIEEntitystandardizationdPrompt(PromptABC):
    template_zh = """
{
    "instruction": "input字段包含用户提供的上下文。命名实体字段包含从上下文中提取的命名实体，这些可能是含义不明的缩写、别名或俚语。为了消除歧义，请尝试根据上下文和您自己的知识提供这些实体的官方名称。请注意，具有相同含义的实体只能有一个官方名称。请按照提供的示例中的输出字段格式，以单个JSONArray字符串形式回复，无需任何解释。并且只提供和香港相关的官方名称",
    "example": {
        "input": "啟鑽商場酒樓冒煙消防救熄 疑抽煙系統失靈，鑽石山彩虹道啟鑽苑啟鑽商場一樓，今（8日）中午12時47分，一間中式酒樓廚房冒煙，職員報案求助，消防接報到場，開喉救熄，事件中無人受傷，毋須疏散。經調查後，相信是油炸食物時抽煙系統失靈，並無可疑。",
        "named_entities": [
            {"name": "消防", "category": "Organization"},
            {"name": "鑽石山彩虹道啟鑽苑啟鑽商場", "category": "Organization"}
        ],
        "output": [
            {"name": "消防", "category": "Organization", "official_name": "香港消防處"},
            {"name": "鑽石山彩虹道啟鑽苑啟鑽商場", "category": "Other", "official_name": "Kai Chuen Shopping Centre_Brochure"}
        ]
    },
    "input": $input,
    "named_entities": $named_entities,
}    
    """
    template_en = template_zh
    @property
    def template_variables(self) -> List[str]:
        return ["input", "named_entities"]

    def parse_response(self, response: str, **kwargs):
        rsp = response
        if isinstance(rsp, str):
            rsp = json.loads(rsp)
        if isinstance(rsp, dict) and "output" in rsp:
            rsp = rsp["output"]
        if isinstance(rsp, dict) and "named_entities" in rsp:
            standardized_entity = rsp["named_entities"]
        else:
            standardized_entity = rsp
        entities_with_offical_name = set()
        merged = []
        entities = kwargs.get("named_entities", [])
        for entity in standardized_entity:
            merged.append(entity)
            entities_with_offical_name.add(entity["name"])
        # in case llm ignores some entities
        for entity in entities:
            if entity["name"] not in entities_with_offical_name:
                entity["official_name"] = entity["name"]
                merged.append(entity)
        return merged
