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
        "instruction": "您是一位专门从事开放信息提取（OpenIE）的专家。请从input字段的文本中提取任何可能的关系（包括主语、谓语、宾语），并按照JSON格式列出它们，须遵循example字段的示例格式。请注意以下要求：1. 每个三元组应至少包含entity_list实体列表中的一个，但最好是两个命名实体。2. 明确地将代词解析为特定名称，以保持清晰度。3.尽可能为每个实体都抽取一定的关系",
        "entity_list": $entity_list,
        "input": "$input",
        "example": {
            "input": "特朗普宣布美国将再次退出《巴黎协定》，来源：新华网 | 2025年01月21日 09:12:47。原标题：特朗普宣布美国将再次退出《巴黎协定》新华社华盛顿1月20日电（记者刘亚南 邓仙来）美国总统特朗普20日签署行政令，宣布美国将再次退出旨在应对气候变化的《巴黎协定》。2015年，联合国气候变化大会达成《巴黎协定》，成为全球应对气候变化的重要成果。2017年6月，时任美国总统特朗普宣布美国将退出《巴黎协定》。2020年11月4日，美国正式退出该协定。此举遭到美国国内和国际社会的广泛批评。2021年1月20日，拜登就任总统首日签署行政令，宣布美国将重新加入《巴黎协定》。同年2月19日，美国正式重新加入《巴黎协定》。",
            "entity_list": [
                {"name": "刘亚楠", "category": "Person"},
                {"name": "邓仙来", "category": "Person"},
                {"name": "特朗普", "category": "Person"},
                {"name": "联合国气候变化大会", "category": "Activity"},
                {"name": "拜登", "category": "Person"},
                {"name": "特朗普宣布美国将再次退出《巴黎协定》", "category": "NewsArticale"}
            ],
            "output":[
                ["刘亚楠", "撰写", "特朗普宣布美国将再次退出《巴黎协定》"],
                ["邓仙来", "撰写", "特朗普宣布美国将再次退出《巴黎协定》"],
                ["特朗普", "退出", "《巴黎协定》"],
                ["拜登", "加入", "《巴黎协定》"]
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
