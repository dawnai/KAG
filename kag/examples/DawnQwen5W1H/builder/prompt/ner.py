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
        "instruction": "你是命名实体识别的专家。现在你需要从香港新闻媒体中提取与模式定义匹配的实体。如果不存在该类型的实体，请返回一个空列表。你需要重点抽取事件实体，因为本次项目属于舆情项目。请注意！部分新闻可能带有刑事案件，这属于重大社会事故，需要重点抽取！请以JSON字符串格式回应。你可以参照example进行抽取",
        "schema": $schema,
        "example": [
            {
                "input": "消拯搜救人员于8日在河中寻获45岁的巫裔男子遗体。又一宗坠河溺毙案，这已是今年开年以来,亚罗士打第3宗坠河溺毙事件。该坠河溺毙事件是于周三（8日）在亚罗士打甘榜哥里基斯拿督坤峇路一带的河流发生，死者是约45岁的巫裔男子。吉打州消拯局高级主任阿末阿米努丁指出，消拯局于周三下午3时04分接获有一名男子坠河后下落不明，派员到场展开搜寻行动。消拯人员把死者遗体抬上车以便送往亚罗士打太平间。他说，搜救人员于当天下午6时42分潜入人中寻人，直到下午6时58分寻获死者，遗体交给警方处理，搜寻行动于晚上7时25分结束行动。今年开年隔天即1月2日，在太子路过港海墘街蓝卓公附近的河边，发生一起28岁华青黄伟宏疑癫痫症发作，脱掉上衣与裤子只身穿一条内裤往草丛的河边跑去而失去踪影，直到隔天（1月3日） 上午9时51分，在距离400米处的丹绒查里河畔处寻获其遗体搁浅在河岸旁。第2宗坠河溺毙案是发生在1月5日，一名45岁印裔男子于当天中午约12时，在米都拉惹路桥的丹绒查理河畔坠河，死者遗体于当天下午4时10分寻获。",
                "output": [
                            {
                                "name": "消拯搜救人员于8日在河中寻获45岁的巫裔男子遗体",
                                "type": "JiuLongEvent",
                                "category": "JiuLongEvent",
                                "description": "一宗坠河溺毙案"
                            },
                            {
                                "name": "港海墘街蓝卓公附近的河边，发生一起28岁华青黄伟宏疑癫痫症发作",
                                "type": "JiuLongEvent",
                                "category": "JiuLongEvent",
                                "description": "另一宗案件"
                            },
                            {
                                "name": "一名45岁印裔男子于当天中午约12时，在米都拉惹路桥的丹绒查理河畔坠河",
                                "type": "JiuLongEvent",
                                "category": "JiuLongEvent",
                                "description": "第二宗坠河溺毙案"
                            },
                            {
                                "name": "周三（8日）",
                                "type": "Date",
                                "category": "Date",
                                "description": "坠河溺毙案事件发生时间"
                            },
                            {
                                "name": "亚罗士打",
                                "type": "Location",
                                "category": "Location",
                                "description": "亚罗士打发生3宗坠河溺毙事件，该坠河溺毙事件是于周三（8日）在亚罗士打甘榜哥里基斯拿督坤峇路一带的河流发生"
                            },
                            {
                                "name": "亚罗士打甘榜哥里基斯拿督坤峇路",
                                "type": "Location",
                                "category": "Location",
                                "description": "坠河溺毙案发生的详细地址"
                            },
                            {
                                "name": "巫裔男子",
                                "type": "Person",
                                "category": "Person",
                                "description": "坠河溺毙案发生的受害者"
                            },
                            {
                                "name": "吉打州消拯局",
                                "type": "Organization",
                                "category": "Organization",
                                "description": "负责调查坠河溺毙案的组织"
                            },
                            {
                                "name": "阿末阿米努丁",
                                "type": "Person",
                                "category": "Person",
                                "description": "吉打州消拯局高级主任,负责调查此事件"
                            },                          
                            {
                                "name": "周三下午3时04分",
                                "type": "Date",
                                "category": "Date",
                                "description": "报警时间"
                            },
                            {
                                "name": "下午6时42分",
                                "type": "Date",
                                "category": "Date",
                                "description": "开始潜入水中找人"
                            },
                            {
                                "name": "晚上7时25分",
                                "type": "Date",
                                "category": "Date",
                                "description": "搜寻结束"
                            },
                            {
                                "name": "1月2日",
                                "type": "Date",
                                "category": "Date",
                                "description": "另一宗案件发生时间，华青黄伟宏疑癫痫症发作"
                            },
                            {
                                "name": "港海墘街蓝卓公",
                                "type": "Location",
                                "category": "Location",
                                "description": "华青黄伟宏疑癫痫症发作地点"
                            },
                            {
                                "name": "（1月3日） 上午9时51分",
                                "type": "Date",
                                "category": "Date",
                                "description": "华青黄伟宏遗体被发现时间"
                            },
                            {
                                "name": "距离400米处的丹绒查里河畔处",
                                "type": "Location",
                                "category": "Location",
                                "description": "华青黄伟宏遗体被发现地点"
                            },
                            {
                                "name": "1月5日",
                                "type": "Date",
                                "category": "Date",
                                "description": "第二宗坠河溺毙案发生时间"
                            },
                            {
                                "name": "45岁印裔男子",
                                "type": "Person",
                                "category": "Person",
                                "description": "第二宗坠河溺毙案的受害者"
                            },
                            {
                                "name": "米都拉惹路桥的丹绒查理河畔",
                                "type": "Location",
                                "category": "Location",
                                "description": "第二宗坠河溺毙案的发生地"
                            },
                            {
                                "name": "下午4时10分",
                                "type": "Date",
                                "category": "Date",
                                "description": "第二宗坠河溺毙案受害人遗体被发现时间"
                            },

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
