# KAG相关问题



### 1、表示模型和生成模型有什么区别？

感觉是指代openie的信息抽取模型和知识库问答模型

### 2、KAG-Builder和SPG-Builder有什么区别？

**SPG-Builder知识构建**：

- 支持结构化和非结构化知识导入。
- 与大数据架构兼容衔接，提供了知识构建算子框架，实现从数据到知识的转换。
- 抽象了知识加工SDK框架，提供实体链指、概念标化和实体归一等算子能力，结合自然语言处理(Natural Language Processing, NLP)和深度学习算法，提高单个类型(Class)中不同实例(Instance)的唯一性水平，支持领域图谱的持续迭代演化

**KAG-Builder知识构建：**

kg-builder 实现了一种对大型语言模型（LLM）友好的知识表示，在 DIKW（数据、信息、知识和智慧）的层次结构基础上，升级 SPG 知识表示能力，在同一知识类型（如实体类型、事件类型）上兼容无 schema 约束的信息提取和有 schema 约束的专业知识构建，并支持图结构与原始文本块之间的互索引表示，为推理问答阶段的高效检索提供支持

所以，KAG-Builder是SPG-Builder的升级版，在SPG的基础上引入了对大模型更友好的知识表示。

KAG-Builder采用的知识表示如下：

<img src="C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\image-20250115111216992.png" alt="image-20250115111216992" style="zoom:50%;" />

私域知识库场景，非结构化数据、结构化信息、业务专家经验 往往三者共存，KAG 参考了 DIKW 层次结构，**将 SPG 升级为对 LLM 友好的版本**。针对新闻、事件、日志、书籍等非结构化数据，交易、统计、审批等结构化数据，业务经验、领域知识等规则，KAG 采用版面分析、知识抽取、属性标化、语义对齐等技术，将原始的业务数据&专家规则融合到统一的业务知识图谱中。



### 3、KAG的信息抽取效果怎么判断？

历史难题咯，这个得根据业务情况具体判断

### 4、bge-m3在KAG中的哪个流程使用？

通过阅读KAG论文发现一下原文：

> [!TIP]
>
> In naive RAG, retrieval is achieved by calculating the similarity (e.g. cosine similarity) between the embeddings of the question and document chunks, where the semantic representation capability of embedding models plays a key role. This mainly includes a sparse encoder (BM25) and a dense retriever (BERT architecture pre-training language models)
>
> 在传统RAG中，检索是通过计算**问题和文档块**的嵌入之间的相似度（如余弦相似度）来实现的，其中嵌入模型的语义表示能力起着关键作用。这主要包括稀疏编码器（BM25）和密集检索器（预训练语言模型的 BERT 架构）

可以发现，BGE-M3大模型是用于知识问答的，作用是将用户输入的query转化为embedding 向量，方便后续做向量相似度匹配。

但是这也意味着知识图谱中也应该有**知识的向量存储**，否则也无法进行高效匹配，查看0.5版本的文档（https://openspg.yuque.com/ndx6g9/0.5/yofw66sq5ncerf4i），发现BGE-M3大模型还要用于**生成实体属性的embedding向量**，这样就可以解释通了。

**源码位置：**

在存储时，调用BGE-M3大模型为实体属性生成embedding向量并进行存储：EmbeddingVectorManager(object)类

C:\Users\dawna\Desktop\KAG\kag\common\graphstore\neo4j_graph_store.py



### 5、为什么在图谱构建阶段有三个大模型prompt文件？

在builder/prompt文件夹下有三个文件：

```
├── __init__.py
├── ner.py
├── std.py
└── triple.py
```

现在依次对三个文件做出解释：

- **ner.py**（用于实体抽取）

ner.py 中定义了**实体抽取的中文、英文模板**，模版以json string 格式呈现，example.input、example.output 分别展示实体抽取阶段大模型的输入示例、输出示例。

- **std.py**（用于实体标准化）

std.py 中定义**实体标准化的中英文模板**，模版以json string 格式呈现，example.input & example.named_entities、example.output 分别展示实体标准化阶段大模型的输入示例、输出示例。

实体标准化依赖大模型对上下文的理解，以及自身的知识储备；标准化的实体名，可补齐实体的上下文，避免歧义。

- **triple.py**（三元组抽取）

triple.py 中定义spo 三元组抽取的中英文模板，模版以json string 格式呈现，example.input & example.entity_list、example.output 分别展示spo 抽取阶段大模型的输入示例、输出示例。

instruction 中要求，spo 抽取结果，其起点 或 终点之一，需要在entity_list 中出现。

图谱构建阶段的流程如下：

1. 实体抽取，使用ner_prompt 
2. 实体标准化，使用std_prompt
3. 三元组抽取，使用triple_prompt
4. 将抽取结果汇总成图结构

```python
class SchemaFreeExtractor(ExtractorABC):
    """
    A class for extracting knowledge graph subgraphs from text using a large language model (LLM).
    Inherits from the Extractor base class.

    Attributes:
        llm (LLMClient): The large language model client used for text processing.
        schema (SchemaClient): The schema client used to load the schema for the project.
        ner_prompt (PromptABC): The prompt used for named entity recognition.
        std_prompt (PromptABC): The prompt used for named entity standardization.
        triple_prompt (PromptABC): The prompt used for triple extraction.
        external_graph (ExternalGraphLoaderABC): The external graph loader used for additional NER.
    """

    def __init__(
        self,
        llm: LLMClient,
        ner_prompt: PromptABC = None,
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

```



# KAG工作日志

### 1月14日：

1. 在自己电脑上成功部署OpenSPG，fork了KAG源码进行初步分析准备。
2. 何老师推荐了一些知识图谱文章，阅读这些文章。
3. 对知识图谱的基础知识有了一定了解，做了丰富的笔记记录。

### 1月15日：

- 查看KAG文档，对比学习SPG的知识图谱和传统知识图谱有哪些区别。
- 下午停电耽搁了
- 在windows上部署ollama，显卡是3070，只能跑1B左右的大模型
- 在开发者模式下开始初始化项目
- 注册硅基流动账户，申请API_key
- example_config.yaml文件始终报编码错：

> [!CAUTION]
>
> UnicodeDecodeError: 'gbk' codec can't decode byte 0xac in position 415: illegal multibyte sequence
>
> debug半天才发现是自己写了中文注释！

然后成功执行第一个抽取任务：

![image-20250115211434589](C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\image-20250115211434589.png)

抽取细节：

```
{"id": "d01b2447d15fae313a7233d11ee180a685f5334aff619c97a502bee8886f1579", "value": {"abstract": {"id": "David Eagl", "name": "David Eagl", "content": "Introducti"}, "graph_stat": {"num_nodes": 26, "num_edges": 37, "num_subgraphs": 1}}}
{"id": "95a37c215180e2f2c37fb7551a997b4643f2c5cc87d2d9d44fef36b1d0ce70bf", "value": {"abstract": {"id": "Karl Deiss", "name": "Karl Deiss", "content": "Introducti"}, "graph_stat": {"num_nodes": 26, "num_edges": 37, "num_subgraphs": 1}}}
{"id": "14a64a641b2f33ccbdb82b88eff9d47ff0cd87ca4549e257e1a6ed0ef65b7884", "value": {"abstract": {"id": "Thomas C. ", "name": "Thomas C. ", "content": "Introducti"}, "graph_stat": {"num_nodes": 34, "num_edges": 49, "num_subgraphs": 1}}}
```

抽取结果：

<img src="C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\image-20250115212448506.png" alt="image-20250115212448506" style="zoom:50%;" />

### 1月16日：

1. 去暨南大学参加保密谈话
2. 李老师给了两本关于知识图谱的书籍，后续阅读一下

### 1月17日：

- 早上看了一会《工业级知识图谱方法与实践》，了解了一些关于工业级知识图谱的相关知识，重点 看了知识融合章节 
- 解决bge-m3在哪里调用的问题。（在知识推理和图谱构建都要使用） 
- 发现项目名称只能大写开头以及字母和数字组成（后期可以改）。 
- 开始实验用bge-large-en-v1.5进行向量嵌入，因为业务重点抽取英文文档，采用英文bge也许效果会 好一些（没卵用，因为上下文字符有限制）。
- 做实验的时候需要删除之前的ckpt日志文件，否则报错。

### 1月18日：

华农不愧是停水停电大学（停电停水一天）

### 1月20日：

- 吸取前两次经验，prompt的定义可能并不是很好，因为后面发现知识图谱构建阶段有三个prompt文件，我只是更改了其中一个triple.py文件，今天对三个文件的作用进行解释说明：


```
├── __init__.py
├── ner.py（用于实体抽取）
├── std.py（用于实体标准化）
└── triple.py（三元组抽取）
```

- 学习了官方prompt使用方式，才发现自己之前效果比较差是因为prompt文件根本就没起作用！（图谱构建运行脚本里面的prompt路径没有更改！）
- 了解了KAG的schema，有点没看懂（自己关于图谱项目经验不多）
- 学习了KAG_schema的基础语法、语法结构、怎么定义实体类型、概念类型、事件类型。

### 1月21日：

- 从何老师那里获取到了自定义schema,自己再新增了一些，得到以下内容：（**记住第一个首字母大写，否则报错**）

```yaml
namespace DawnBgeTest1

Chunk(文本块):EntityType
    properties:
        content(内容):Text
            index:TextAndVector

Activity(活动会议):EntityType
    properties:
        desc(描述):Text
        nameEn(名称):Text

Location(地点):EntityType
    properties:
        desc(描述):Text
        nameEn(名字):Text

Date(日期): EntityType
     properties:
        desc(描述): Text
            index: TextAndVector
        semanticType(语义类型): Text
            index: Text

NewsArticale(新闻文章):EntityType
    properties:
        summary(摘要):Text
        author(作者):Text
        title(标题):Text
        publishTime(发布时间):Text

Others(其他):EntityType
    properties:
        desc(描述):Text
            index:TextAndVector
        semanticType(语义类型):Text
            index:Text

Person(人物): EntityType
     properties:
        desc(描述): Text
        nameEn(名字):Text
        age(年龄):Text
        participants(参与):Activity
            constraint:MultiValue
            properties:
                time(时间):Text
        writer(撰写):NewsArticale
            constraint:MultiValue
            properties:
                publishTime(出版时间):Text
        appear(出现):location
            constraint:MultiValue
            properties:
                time(时间):Text
```

Schema可视化展示如下：

<img src="C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\image-20250121161924894.png" alt="image-20250121161924894" style="zoom:50%;" />



现在的实体类型就精简很多了，只是针对人物信息进行抽取，相应的，我再将ner.py文件的prompt再改成人物相关的提示：

```
    template_zh = """
    {
        "instruction": "你是命名实体识别的专家。请从输入中提取与模式定义匹配的实体。如果不存在该类型的实体，请返回一个空列表。请以JSON字符串格式回应。你可以参照example进行抽取。",
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
                        {"entity": "特朗普宣布美国将再次退出《巴黎协定》", "category": "NewsArticale"},
                        {"entity": "2025年01月21日", "category": "Date"},
                        {"entity": "2020年11月4日", "category": "Date"},
                        {"entity": "2021年1月20日", "category": "Date"},
                        {"entity": "2017年6月", "category": "Date"}
                    ]
            }
        ],
        "input": "$input"
    }    
        """
```

于此相对应的triple.py文件内容如下：

```
template_zh = """
{
    "instruction": "您是一位专门从事开放信息提取（OpenIE）的专家。请从input字段的文本中提取任何可能的关系（包括主语、谓语、宾语），并按照JSON格式列出它们，须遵循example字段的示例格式。请注意以下要求：1. 每个三元组应至少包含entity_list实体列表中的一个，但最好是两个命名实体。2. 明确地将代词解析为特定名称，以保持清晰度。",
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
            {"name": "特朗普宣布美国将再次退出《巴黎协定》", "category": "NewsArticale"},
            {"name": "2025年01月21日", "category": "Date"},
            {"name": "2020年11月4日", "category": "Date"},
            {"name": "2021年1月20日", "category": "Date"},
            {"name": "2017年6月", "category": "Date"},
        ],
        "output":[
            ["刘亚楠", "撰写", "特朗普宣布美国将再次退出《巴黎协定》"],
            ["邓仙来", "撰写", "特朗普宣布美国将再次退出《巴黎协定》"],
            ["特朗普", "退出", "《巴黎协定》"],
            ["拜登", "加入", "《巴黎协定》"],
            ["2017年6月", "退出", "《巴黎协定》"],
            ["2021年1月20日", "加入", "《巴黎协定"]
        ]
    }
}    
    """
```

待抽取数据如下：

```
特朗普：将在上任后的“几个小时内”推翻拜登多项行政令
来源：新华网 | 2025年01月20日 09:34:55
原标题：特朗普：将在上任后的“几个小时内”推翻拜登多项行政令
　　新华社华盛顿1月19日电（记者熊茂伶）美国候任总统特朗普19日说，他将在20日宣誓就职后的“几个小时内”推翻拜登政府的多项行政令。

　　特朗普19日下午在首都华盛顿特区第一资本体育馆举行的集会上说，拜登政府的每一项“激进而愚蠢”的行政命令，在他宣誓就职后的几小时内都会被废除。他还称，自己的上任将结束“美国衰退的四年”。

　　特朗普表示，他即将签署的行政命令涉及边境安全、能源、联邦政府开支、短视频社交媒体平台TikTok、“多元化、公平与包容”政策等。

　　据美国多家媒体分析，特朗普将在就职当天签署超过100项行政命令。

　　特朗普将于20日中午宣誓就职。由于首都华盛顿特区预计出现严寒天气，他的就职典礼将改为在室内举行。
```

> [!CAUTION]
>
> 报错：
>
> UnicodeDecodeError: 'gbk' codec can't decode byte 0xae in position 36: illegal multibyte sequence
>
> 还是编码错误，因为使用中文原因，在C:\Users\dawna\Desktop\KAG\kag\builder\component\scanner\dataset_scanner.py中的110行指定utf-8：
>
> with open(*input*, "r",*encoding*='utf-8') as f:
>
> 何老师这个坑已经帮我踩了，哈哈哈

抽取结果：

```
{"id": "73280896e15e86572970e120401afc1e0f5078b0d55e48e1365d5ef2daf3a800", "value": {"abstract": {"id": "�����գ��������κ��", "name": "�����գ��������κ��", "content": "�»��绪ʢ��1��19"}, "graph_stat": {"num_nodes": 15, "num_edges": 21, "num_subgraphs": 1}}}
```

出现了乱码，但这是正常的，因为日志文件是输出不了中文。可以看到实体抽取了15个，边只有21个，这大大降低了知识图谱的冗余。

去neo4j看看：

![graph](C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\graph.png)

整个图谱精简了很多，但是有明显的错误，为什么抽取成2021年？明明是2025年啊？

怀疑是Schema的时间定义有问题，因为后续并没有继承这个实体（我自己定义的时间）。

- **在Schema中删除Date 实体，在prompt中删除相应的prompt，后再抽取一遍：**

![graph (1)](C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\graph (1).png)

时间这次是正确的，但是没有了年份，感觉也不是特别好。

有一个想法，大模型本身就自带很多知识，为什么一定要严格按照input文本进行抽取呢？让大模型适当发挥一下其自身的知识库，做一个补充会不会更好？

但是仔细想一想，发现不对，因为KAG后面还有知识问答大模型，到时候让那个大模型参考参考自己的知识库不就好了？builder这一阶段严格抽取input信息就行了。

下一阶段任务就是丰富Schema，然后配套更新prompt。之后就要看reasoner、solver阶段的源码。
