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
from typing import Optional, List

from kag.interface import PromptABC


@PromptABC.register("example_medical_triple")
class OpenIETriplePrompt(PromptABC):
    template_zh = """
{
    "instruction": "您是一位专门从事开放信息提取（OpenIE）的专家。请从input字段的文本中提取任何可能的关系（包括主语、谓语、宾语），并按照JSON格式列出它们，须遵循example字段的示例格式。请注意以下要求：1. 每个三元组应至少包含entity_list实体列表中的一个，但最好是两个命名实体。2. 明确地将代词解析为特定名称，以保持清晰度。3.尽可能的保持简洁，对于那些重复冗余的实体请不要抽取。比如同义词，同一种类实体的不同表达方式等。4.输入都是英文，但是要求输出的json里面都是中文",
    "entity_list": $entity_list,
    "input": "$input",
    "example": {
        "input": ""Twin Peaks film director David Lynch dies at 78.\n\nDavid Lynch, the American filmmaker whose works include the surrealist cult classics Mulholland Drive and Twin Peaks, has died aged 78.\n\nLynch's death was announced on his official Facebook page by his family on Thursday.\n\n\"There's a big hole in the world now that he's no longer with us,\" the post said.\n\n\"But, as he would say, 'Keep your eye on the donut and not on the hole.'… It's a beautiful day with golden sunshine and blue skies all the way.\"\n\nLynch revealed in August last year he was battling emphysema, a chronic lung disease, from \"many years of smoking\".\n\nConsidered by many a maverick filmmaker, he received three best director Oscar nominations throughout his career for his work on Blue Velvet, The Elephant Man and Mulholland Drive.\n\nHis last major project was Twin Peaks: The Return, which was broadcast in 2017, and continued the TV series that ran for two seasons in the early 1990s.\n\nObituary: Mind-bending director who embraced the weird.\n\n\"David was in tune with the universe and his own imagination on a level that seemed to be the best version of human,\" actor Kyle MacLachlan, who starred in many of Lynch's projects including Twin Peaks, wrote in tribute.\n\n\"He was not interested in answers because he understood that questions are the drive that make us who we are.\"\n\nDescribing Lynch as \"an enigmatic and intuitive man with a creative ocean bursting forth inside of him\", MacLachlan added: \"My world is that much fuller because I knew him and that much emptier now that he's gone.\"\n\nLynch won the prestigious Palme d'Or at the Cannes film festival for Wild at Heart in 1990.\n\n'One of a kind'\n\nThe star of that film, Nicolas Cage, told the BBC World Service's Newshour programme he was one of the main reasons he fell in love with cinema.\n\n\"I used to see his movie Eraserhead in Santa Monica,\" he said. \"He's largely instrumental for why I got into filmmaking. He was one of a kind. He can't be replaced.\"\n\nFellow film director Steven Spielberg said he was a \"singular, visionary dreamer who directed films that felt handmade\".\n\n\"The world is going to miss such an original and unique voice,\" he added in a statement to Variety.\n\nDirector Ron Howard called him a \"gracious man and fearless artist who followed his heart & soul proved that radical experimentation could yield unforgettable cinema\".\n\nMusician Moby, for whom Lynch directed the video for Shot In The Back Of The Head, said he was \"just heartbroken\".\n\nMany of Lynch's films were known for their surrealist, dreamlike quality.\n\nEraserhead, his first major release in 1977, was filled with dark, disturbing imagery.\n\n\"While his imagination clearly has an eye for the viscerally potent, this remains an unremarkable feat by his later standards,\" a BBC reviewer said of the film in 2001.\n\nIn a May 2024 interview with BBC Radio Three's Sound of Cinema, Lynch described the process of working with late composer Angelo Badalamenti, who designed many of the soundscapes that accompanied his vision.\n\n\"And then I say, 'no that's still too fast, it's not dark enough, it's not heavy and foreboding enough,'\" Lynch recalled.\n\nHis body of work was recognised at the Oscars in 2020 when he was given an honorary Academy Award.\n\nThe director said last year that, despite his emphysema diagnosis, he was in \"excellent shape\" and would \"never retire\".\n\nHe added the diagnosis was the \"price to pay\" for his smoking habit.\n\nBut his condition deteriorated within months. In a November interview with People magazine, he said he needed oxygen to walk.\n\nBorn in Missoula, Montana, Lynch first began a career in painting before switching to making short films during the 1960s."",
        "entity_list": [
            {"entity": "大卫林奇", "category": "person"},
            {"entity": "凯尔·麦克拉克伦", "category": "person"},
            {"entity": "Facebook", "category": "software"},
            {"entity": "肺气肿", "category": "Disease"},
            {"entity": "《蓝丝绒》", "category": "opus"},
            {"entity": "《象人》", "category": "opus"},
            {"entity": "《穆赫兰道》", "category": "opus"},
            {"entity": "《双峰》", "category": "opus"},
            {"entity": "《橡皮头》", "category": "opus"},
            {"entity": "奥斯卡", "category": "reward"},
            {"entity": "戛纳电影节", "category": "reward"},
            {"entity": "蒙大拿州米苏拉", "category": "location"},
            {"entity": "安杰洛·巴达拉曼蒂", "category": "person"},
            {"entity": "莫比 (Moby)", "category": "person"},
            {"entity": "超现实主义", "category": "style"},

        ],
        "output":[
            ["大卫林奇", "去世", "2025年"],
            ["大卫林奇", "第一部作品", "《橡皮头》"],
            ["大卫林奇", "指导", "《双峰》"],
            ["大卫林奇", "被授予", "奥斯卡"],
            ["大卫林奇", "合作", "安杰洛·巴达拉曼蒂"],
            ["大卫林奇", "风格", "超现实主义"],
            ["凯尔·麦克拉克伦", "主演", "《双峰》"],
            ["大卫林奇", "出生地", "蒙大拿州米苏拉"],
            ["大卫林奇", "获得", "戛纳电影节"],
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
