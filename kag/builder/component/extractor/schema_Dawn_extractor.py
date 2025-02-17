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
import hanlp
import copy
import logging
from typing import Dict, Type, List

from kag.interface import LLMClient#用于与大型语言模型（LLM）交互。
from tenacity import stop_after_attempt, retry

from kag.interface import ExtractorABC, PromptABC, ExternalGraphLoaderABC

from kag.common.conf import KAG_PROJECT_CONF#项目配置类
from kag.common.utils import processing_phrases, to_camel_case
from kag.builder.model.chunk import Chunk #文本块
from kag.builder.model.sub_graph import SubGraph #子图
from kag.builder.prompt.utils import init_prompt_with_fallback#初始化prompt
from knext.schema.client import OTHER_TYPE, CHUNK_TYPE, BASIC_TYPES
from knext.common.base.runnable import Input, Output
from knext.schema.client import SchemaClient

logger = logging.getLogger(__name__)#初始化日志记录器


@ExtractorABC.register("schema_Dawn")
@ExtractorABC.register("schema_Dawn_extractor")
class SchemaDawnExtractor(ExtractorABC):
    """
    A class for extracting knowledge graph subgraphs from text using a large language model (LLM).
    Inherits from the Extractor base class.

    Attributes:
        llm：用于与大型语言模型交互的客户端。
        schema：用于加载知识图谱模式的模板。
        ner_prompt：用于命名实体识别的提示。
        std_prompt：用于命名实体标准化的提示。
        triple_prompt：用于三元组提取的提示。
        external_graph：用于加载外部图的加载器。
    """

    def __init__(
        self,
        llm: LLMClient,
        ner_prompt: PromptABC = None,#默认值为None
        std_prompt: PromptABC = None,
        triple_prompt: PromptABC = None,
        external_graph: ExternalGraphLoaderABC = None,
    ):
        """
        Initializes the KAGExtractor with the specified parameters.

        Args:
            llm (LLMClient): The large language model client.
            ner_prompt (PromptABC, optional): The prompt for named entity recognition. Defaults to None.
            std_prompt (PromptABC, optional): The prompt for named entity standardization. Defaults to None.
            triple_prompt (PromptABC, optional): The prompt for triple extraction. Defaults to None.
            external_graph (ExternalGraphLoaderABC, optional): The external graph loader. Defaults to None.
        """
        super().__init__()
        self.llm = llm
        self.schema = SchemaClient(project_id=KAG_PROJECT_CONF.project_id).load()#加载项目的知识图谱模板
        self.ner_prompt = ner_prompt
        self.std_prompt = std_prompt
        self.triple_prompt = triple_prompt#三元组prompt，也就是关系prompt

        biz_scene = KAG_PROJECT_CONF.biz_scene#获取项目的业务场景
        if self.ner_prompt is None:
            self.ner_prompt = init_prompt_with_fallback("ner", biz_scene)
        if self.std_prompt is None:
            self.std_prompt = init_prompt_with_fallback("std", biz_scene)
        if self.triple_prompt is None:
            self.triple_prompt = init_prompt_with_fallback("triple", biz_scene)

        self.external_graph = external_graph

    @property
    def input_types(self) -> Type[Input]:#输入类型为文本块
        return Chunk

    @property
    def output_types(self) -> Type[Output]:#输出类型为知识图谱子图
        return SubGraph
    #格式转换器
    def convert_to_new_format(self,data):
        """
        将原始格式 [[('Pema', 'ORGANIZATION', 5, 6), ('Khandu', 'ORGANIZATION', 6, 7), ...]]
        转换为 [{'name': 'Pema', 'type': 'ORGANIZATION', 'category': 'ORGANIZATION', 'description': ''}, ...]
        """
        # 确保输入是列表的列表
        if not data or not isinstance(data[0], list):
            raise ValueError("输入数据格式不正确，应为嵌套列表形式")

        # 提取内部列表
        entities = data[0]

        # 转换为新格式
        result = []
        for entity in entities:
            name, entity_type, start, end = entity  # 解包元组
            result.append({
                'name': name,
                'type': entity_type,
                'category': entity_type,
                'description': ''
            })

        return result
    @retry(stop=stop_after_attempt(3))#使用重试机制，最多重试 3 次，函数执行失败时自动重试
    def named_entity_recognition(self, passage: str):
        """
        Performs named entity recognition on a given text passage.
        Args:
            passage (str): 输入文本段落.
        Returns:
            合并去重后的实体列表，每个实体包含名称、类型、类别和描述.
        """
        """
        LLM返回示例:[{'name':'','type':'','category':'','description':''},{'name':'','type':'','category':'','description':''}]
        hanlp返回示例:[[('Pema', 'ORGANIZATION', 5, 6), ('Khandu', 'ORGANIZATION', 6, 7), ('PTI', 'ORGANIZATION', 10, 11)]]
        """
        
        # 语种见名称最后一个字段或相应语料库
        tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        ner = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH)
        tok_out=tok([passage])
        ner_out=ner(tok_out)
        #将Hanlp返回格式变成标准格式
        ner_result = self.convert_to_new_format(ner_out)
        print(ner_result)
        # ner_result = self.llm.invoke({"input": passage}, self.ner_prompt)#传入passage和prompt
        
        #通过向量查询执行匹配，允许根据查询向量和图数据中的节点进行相似度匹配。结果会根据 score 值进行过滤，只有分数大于等于阈值（threshold）的结果才会被返回
        if self.external_graph: #调用外部图补充实体
            extra_ner_result = self.external_graph.ner(passage)
        else:
            extra_ner_result = []

        #处理外部图返回的实体，外部图优先级更高，
        output = []
        dedup = set()#使用 dedup 集合记录已处理的实体名称，避免重复添加,但是这种去重是严格按照字符进行去重，并不合理。
        for item in extra_ner_result:
            name = item.name
            label = item.label
            description = item.properties.get("desc", "")
            semantic_type = item.properties.get("semanticType", label)
            if name not in dedup:
                dedup.add(name)
                output.append(
                    {
                        "name": name,
                        "type": semantic_type,
                        "category": label,
                        "description": description,
                    }
                )
        #处理LLM返回的实体
        for item in ner_result:
            name = item.get("name", None)
            if name and name not in dedup:
                dedup.add(name)
                output.append(item)
        
        return output

    @retry(stop=stop_after_attempt(3))#失败后重试3次，实体标准化部分
    def named_entity_standardization(self, passage: str, entities: List[Dict]):
        """
        Standardizes named entities.

        Args:
            passage (str): 输入的文本块.
            entities (List[Dict]): 实体列表.

        Returns:
            Standardized entity information.
        """
        # 直接将大模型的回答进行返回
        return self.llm.invoke(
            {"input": passage, "named_entities": entities}, self.std_prompt
        )

    @retry(stop=stop_after_attempt(3))
    def triples_extraction(self, passage: str, entities: List[Dict]):
        """
        Extracts triples (subject-predicate-object structures) from a given text passage based on identified entities.
        Args:
            passage (str): 输入的文本块.
            entities (List[Dict]): 实体列表.
        Returns:
            The result of the triples extraction operation.
        """
        return self.llm.invoke( #还是直接返回抽取到的关系
            {"input": passage, "entity_list": entities}, self.triple_prompt
        )

    def assemble_sub_graph_with_spg_records(self, entities: List[Dict]):
        """
        根据实体列表（entities）构建一个知识图谱的子图（SubGraph）。
        通过解析每个实体的属性，并根据这些属性与知识图谱模式（schema）的定义来添加节点和边.

        Args:
            entities (List[Dict]): A list of entities to be used for subgraph assembly.

        Returns:
            制作好的子图 and  更新后的实体列表.
        """
        sub_graph = SubGraph([], [])#初始化一个空的子图对象 SubGraph
        for record in entities:
            s_name = record.get("name", "")#实体列表中获取实体名称
            s_label = record.get("category", "")#实体列表中获取标签
            properties = record.get("properties", {})#从实体列表中获取属性
            tmp_properties = copy.deepcopy(properties)#创建属性的深拷贝，以免修改原始属性
            spg_type = self.schema.get(s_label)#根据实体的类别（s_label），从知识图谱模式（self.schema）中获取对应的模式类型（spg_type）

            for prop_name, prop_value in properties.items():#遍历实体的每个属性

                if prop_value == "NAN":
                    tmp_properties.pop(prop_name)#从 tmp_properties 中移除该属性
                    continue

                if prop_name in spg_type.properties:#属性名在Schema中定义了
                    from knext.schema.model.property import Property

                    prop: Property = spg_type.properties.get(prop_name)
                    o_label = prop.object_type_name_en#获取属性的目标类型（o_label），即该属性指向的节点类型
                    if o_label not in BASIC_TYPES:#如果目标类型不是基本类型（BASIC_TYPES），说明它指向的是另一个实体

                        if isinstance(prop_value, str):#如果属性值是字符串，将其转换为列表（以支持多值属性）
                            prop_value = [prop_value]

                        for o_name in prop_value:#遍历每个属性值（o_name）
                            sub_graph.add_node(id=o_name, name=o_name, label=o_label)#添加节点
                            sub_graph.add_edge(#添加边
                                s_id=s_name,
                                s_label=s_label,
                                p=prop_name,
                                o_id=o_name,
                                o_label=o_label,
                            )
                        tmp_properties.pop(prop_name)#从 tmp_properties 中移除已处理的属性，避免将其作为节点的属性存储
            record["properties"] = tmp_properties#将处理后的属性（tmp_properties）更新回实体记录（record）
            sub_graph.add_node(#在子图中添加一个节点，表示当前实体
                id=s_name, name=s_name, label=s_label, properties=properties
            )
        return sub_graph, entities#返回子图和实体列表

    @staticmethod
    def assemble_sub_graph_with_triples(
        sub_graph: SubGraph, entities: List[Dict], triples: List[list]
    ):
        """
        主要功能是根据三元组列表（triples）和实体列表（entities）构建子图（SubGraph）中的边
        但是会产生大量other_type实体！
        
        Args:
            sub_graph (SubGraph): The subgraph to add edges to.
            entities (List[Dict]): A list of entities, for looking up category information.
            triples (List[list]): A list of triples, each representing a relationship to be added to the subgraph.
        Returns:
            返回子图.

        """

        def get_category(entities_data, entity_name):
            for entity in entities_data:
                if entity["name"] == entity_name:
                    return entity["category"] #返回实体类别
            return None

        for tri in triples:#遍历每个三元组，跳过不符合格式的三元组
            if len(tri) != 3:
                continue

            s_category = get_category(entities, tri[0])#获取主体类别

            tri[0] = processing_phrases(tri[0])#主体名称进行预处理（转换为字符串、转化为小写字母、去除特殊字符，只保留字母、数字、汉字和空格、清理首尾空格）
            if s_category is None:
                s_category = OTHER_TYPE#如果主体类别为空，设置为OTHER_TYPE
                sub_graph.add_node(tri[0], tri[0], s_category)

            o_category = get_category(entities, tri[2])#获取客体类别
            tri[2] = processing_phrases(tri[2])
            if o_category is None:
                o_category = OTHER_TYPE
                sub_graph.add_node(tri[2], tri[2], o_category)
            #添加边
            edge_type = to_camel_case(tri[1])
            if edge_type:
                sub_graph.add_edge(tri[0], s_category, edge_type, tri[2], o_category)

        return sub_graph

    @staticmethod
    def assemble_sub_graph_with_chunk(sub_graph: SubGraph, chunk: Chunk):
        """
        其主要功能是将一个 Chunk 对象（包含文本和元数据）与知识图谱的子图（SubGraph）关联起来，基本是一个新闻一个chunk，如果新闻长度比较短，可能一个chunk包含多个新闻。
        该函数的核心任务是将 Chunk 对象作为一个节点添加到子图中，并将该节点与子图中现有的所有节点建立连接（注意，是所有！）
        Args:
            sub_graph (SubGraph): The subgraph to add the chunk information to.
            chunk (Chunk): The chunk object containing the text and metadata.
        Returns:
            The constructed subgraph.
        """
        for node in sub_graph.nodes:# 遍历子图中的所有节点并添加边
            sub_graph.add_edge(node.id, node.label, "source", chunk.id, CHUNK_TYPE)

        sub_graph.add_node(#将Chunk 对象作为一个节点添加到子图中
            chunk.id,
            chunk.name,
            CHUNK_TYPE,
            {
                "id": chunk.id,
                "name": chunk.name,
                "content": f"{chunk.name}\n{chunk.content}",
                **chunk.kwargs,
            },
        )
        sub_graph.id = chunk.id
        return sub_graph

    def assemble_sub_graph_with_entities(
        self, sub_graph: SubGraph, entities: List[Dict]
    ):
        """
        主要功能是根据命名实体（entities）信息构建子图（SubGraph）

        Args:
            sub_graph (SubGraph): The subgraph object to be assembled.
            entities (List[Dict]): A list containing entity information.
        没有返回，（直接修改传入的 sub_graph 对象）
        """

        for ent in entities:#遍历实体列表
            name = processing_phrases(ent["name"])#进行预处理，转小写等等
            sub_graph.add_node(#将实体作为节点添加到子图中
                name,
                name,
                ent["category"],
                {
                    "desc": ent.get("description", ""),#实体的描述（如果存在）
                    "semanticType": ent.get("type", ""),#实体的语义类型（如果存在）
                    **ent.get("properties", {}),#将实体的其他属性展开并作为节点的属性
                },
            )

            if "official_name" in ent:#如果有官方名称，将官方名称作为独立节点添加，并与普通名称节点建立连接。
                official_name = processing_phrases(ent["official_name"])
                if official_name != name:
                    sub_graph.add_node(
                        official_name,
                        official_name,
                        ent["category"],
                        {
                            "desc": ent.get("description", ""),
                            "semanticType": ent.get("type", ""),
                            **ent.get("properties", {}),
                        },
                    )
                    sub_graph.add_edge(
                        name,
                        ent["category"],
                        "OfficialName",
                        official_name,
                        ent["category"],
                    )
    def assemble_sub_graph(
        self,
        sub_graph: SubGraph,
        chunk: Chunk,
        entities: List[Dict],
        triples: List[list],
    ):
        """
        主要功能是将实体（entities）和三元组（triples）信息整合到一个子图（SubGraph）中，并将子图与一个文本块（Chunk）关联起来.
        最先是用实体构建子图，再用三元组为子图添加边，然后才是将chunk集成到子图
        Args:
            sub_graph (SubGraph): The subgraph to be assembled.
            chunk (Chunk): The chunk of text the subgraph is about.
            entities (List[Dict]): A list of entities identified in the chunk.
            triples (List[list]): A list of triples representing relationships between entities.
        Returns:
            The constructed subgraph.
        """
        self.assemble_sub_graph_with_entities(sub_graph, entities)
        self.assemble_sub_graph_with_triples(sub_graph, entities, triples)
        self.assemble_sub_graph_with_chunk(sub_graph, chunk)
        return sub_graph
    

    def append_official_name(
        self, source_entities: List[Dict], entities_with_official_name: List[Dict]
    ):
        """
        将官方名称（official_name）附加到一组实体（source_entities）中
        Args:
            source_entities (List[Dict]): A list of source entities.
            entities_with_official_name (List[Dict]): A list of entities with official names.
        """
        try:
            tmp_dict = {}
            for tmp_entity in entities_with_official_name:
                if "name" in tmp_entity:
                    name = tmp_entity["name"]
                elif "entity" in tmp_entity:
                    name = tmp_entity["entity"]
                else:
                    continue
                category = tmp_entity["category"]
                official_name = tmp_entity["official_name"]
                key = f"{category}{name}"
                tmp_dict[key] = official_name

            for tmp_entity in source_entities:
                name = tmp_entity["name"]
                category = tmp_entity["category"]
                key = f"{category}{name}"
                if key in tmp_dict:
                    official_name = tmp_dict[key]
                    tmp_entity["official_name"] = official_name
        except Exception as e:
            logger.warn(f"failed to process official name, info: {e}")


    def _invoke(self, input: Input, **kwargs) -> List[Output]:
        """
        Invokes the semantic extractor to process input data.

        Args:
            input (Input): Input data containing name and content.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Output]: 返回一个列表，包含处理结果（List[Output]），其中每个结果是一个子图（SubGraph）。
        """
        #将标题和内容拼接成一个完整的文本段落（passage），用于后续处理。
        title = input.name
        passage = title + "\n" + input.content

        out = []
        entities = self.named_entity_recognition(passage)#使用大模型提取命名实体
        sub_graph, entities = self.assemble_sub_graph_with_spg_records(entities)#将提取的实体组装成一个子图（sub_graph），但是要使用SPG记录，所以每次试验都要删除ckpt日志文件
        
        #遍历 entities 列表，提取每个实体的 "name" 和 "category" 属性，生成一个新的实体列表（filtered_entities）。
        filtered_entities = [
            {k: v for k, v in ent.items() if k in ["name", "category"]}
            for ent in entities
        ]
        triples = self.triples_extraction(passage, filtered_entities)#调用大模型进行三元组提取
        std_entities = self.named_entity_standardization(passage, filtered_entities)#调用大模型进行实体标准化
        self.append_official_name(entities, std_entities)#为实体列表添加official_name
        self.assemble_sub_graph(sub_graph, input, entities, triples)
        out.append(sub_graph)
        return out
