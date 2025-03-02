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

### 2月6号

新的一年开始，开工大吉！

首先对去年工作内容进行稳固加强，查看工作日志，了解详细情况，以便后续工作展开

由于新买了mac book air，所以将SPG迁移到macbook上，让其当作服务器使用（校园网没权限指定IP地址，这比较烦）。

在创建知识库的时候遇到问题：

```
报错：Done initialize project config with host addr http://127.0.0.1:8887 and project_id 1
No config found.
Error: invalid vectorizer config: Error code: 503 - {'code': 50505, 'message': 'Model service overloaded. Please try again later.', 'data': None}
```

这是硅基流动厂商的问题，服务器不够用了，切换成付费的：Pro/BAAI/bge-m3发现还是不行，所以打算用ollama进行本地调用。

后续使用过程中发现DeepSeek响应太慢，导致超时了，看来需要换一个不热门的大模型了，试一试Kimi

KIMI抽取效果和DeepSeek不相上下（也许是因为schema和抽取样例比较简单的原因）

开始进行实验：在抽取样例固定的情况下比较不同prompt和schema的搭配效果。

### 2月7号：

进行了大量prompt实验，感觉自己是个废物，兜兜转转还没有默认的好用！

kimi需要收费才能使用了（氪金）

### 2月8号：

持续推进prompt实验，将实验方向改为香港舆情监控，模型更改为Qwen2.5 72B 采用阿里云API

想把实体抽取偏向限制在香港地区，但是失败。

### 2月10号：

1、查看源码，寻找大模型实体抽取部分的源码。

2、在Mac上部署Dify，总是遇到镜像拉取问题:

```shell
[+] Running 9/9
 ✘ worker Error                                                            3.2s
 ✘ redis Error                                                             3.2s
 ✘ sandbox Error                                                           3.2s
 ✘ web Error                                                               3.2s
 ✘ db Error                                                                3.2s
 ✘ api Error                                                               3.2s
 ✘ weaviate Error                                                          3.2s
 ✘ ssrf_proxy Error                                                        3.2s
 ✘ nginx Error                                                             3.2s
Error response from daemon: Get "https://registry-1.docker.io/v2/": EOF
```

在mac和windows都遇到了相同的问题。真烦心！防火墙、docker登录、docker权限、国内镜像、梯子，能做的都做了，还是不行。

后面远程朋友的电脑，利用公司的网络拉取，总算是拉下来了，然后用离线镜像命令安装了dify

虽然兜兜转转部署成功了dify，但是后续还要拉镜像岂不是太麻烦？在查看了大量docker的经验贴后，发现需要配置docker engine即可：

```json
{
  "builder": {
    "gc": {
      "defaultKeepStorage": "20GB",
      "enabled": true
    }
  },
  "experimental": false,
  "features": {
    "buildkit": true
  },
  "registry-mirrors": [
    "https://dockerpull.org",
    "https://docker.1panel.dev",
    "https://docker.foreverlink.love",
    "https://docker.fxxk.dedyn.io",
    "https://docker.xn--6oq72ry9d5zx.cn",
    "https://docker.zhai.cm",
    "https://docker.5z5f.com",
    "https://a.ussh.net",
    "https://docker.cloudlayer.icu",
    "https://hub.littlediary.cn",
    "https://hub.crdz.gq",
    "https://docker.unsee.tech",
    "https://docker.kejilion.pro",
    "https://registry.dockermirror.com",
    "https://hub.rat.dev",
    "https://dhub.kubesre.xyz",
    "https://docker.nastool.de",
    "https://docker.udayun.com",
    "https://docker.rainbond.cc",
    "https://hub.geekery.cn",
    "https://docker.1panelproxy.com",
    "https://atomhub.openatom.cn",
    "https://docker.m.daocloud.io",
    "https://docker.1ms.run",
    "https://docker.linkedbus.com"
  ]
}
```

我之间只加了一个镜像，网上这篇攻略几乎把所有国内镜像都加入了，这次就可以顺利拉取了。

3、使用刘博给出的实际新闻数据进行实验

因为是xlsx文件，目前KAG是不支持该文件的读取的，可以改为csv文件，同时修改配置文件kag_config.yaml，将其中的scanner改为csv_scanner.

执行的时候报了一堆错，同时数据太大了，一瞬间就抽取了1k多实体，明天将新闻内容大幅减少，然后再修改报错

### 2月11日

进行了新一轮的KAG prompt实验，得到了新的实验总结：

1. 大模型每次抽取结果具有一定的随机性
2. 对于某些含有暴力血腥类词汇的新闻媒体，大模型会拒绝回答
3. 大模型返回的JSON文件有时候会不符合要求
4. csv文档不一定是最好的xlsx的转换格式，后续还可以试一试其他格式

### 2月12日

去暨大开会，讨论香港九龙项目，同时请教何老师KAG相关问题，根据何老师安排，测试HanLP项目（神经网络的NLP）

### 2月13日

1、部署HanLP开源项目，测试其中的命名实体识别功能

2、进行HanLP多任务命名实体识别实验，测试了8个模型的实体识别效果

3、尝试进行白名单实体识别和黑名单实体识别，成功完成

**实验总结：**

1. HanLP多任务模型可以直接从句子中识别出命名实体，根据实验结果，目前表现比较好的是CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH模型

   **优点：**

   1. 该模型在ONTONOTES词典和msra词典下都有不错的表现
   2. 识别比较稳定，几乎每次结果都一样
   3. 官方虽然没有说支持粤语，但是在实测下，可以进行粤语命名实体识别任务

   **缺点：**

   1. 该模型只能支持中文，项目后续抽取英文的时候，需要额外引入英文版本模型
   2. 多任务模型不支持单个任务的微调。比如多任务模型的分词性能很差，我想对分词器进行一些微调，但这是做不到的

2. 单任务模型灵活性很大，我后期可以自建识别链条，对分词器、命名实体识别、词性标注、语种检测都可以采用不同的模型，但是工作量有些大，因为要结合实际数据搭配不同的模型组合，模型组合可太多了

3. HanLP不支持自定义词典，所有模型都在ONTONOTES词典和msra词典上训练，**这一点是最伤的**。但是支持白名单和黑名单，可以着重抽取关键词。

4. 后期可以将命名实体识别功能集成在KAG中，在实体抽取那一部分替换成HanLP，这样速度会快很多，大模型只需要负责实体对齐和关系抽取即可。

### 2月14日

1、测试HanLP单任务模型的命名实体抽取功能，引入多种分词模型，测试那种搭配效果比较好。

2、COARSE_ELECTRA_SMALL_ZH分词模型和MSRA_NER_ELECTRA_SMALL_ZH实体抽取模型的搭配效果比较好。

### 2月15-2月16（周六周天休息）

### 2月17日

1. 查看KAG实体抽取部分的源码extractor
2. 重写extractor，将上周的COARSE_ELECTRA_SMALL_ZH模型+MSRA_NER_ELECTRA_SMALL_ZH模型集成进KAG，替代原有KAG的实体抽取部分代码。
3. 移植完成后，进行新闻样例的命名实体抽取测试，由于上述两个模型是传统的神经网络模型，所以自定义的Schema不可用，只能用他们之前训练的词典msra，实验结果证明，像新闻这种比较复杂的样例，用传统的NLP效果并不是很好。
4. 对extractor源码进行拆解，撰写extractor函数详细说明：

![image-20250217170207977](./assets/image-20250217170207977.png)

5、源码拆分完后，发现了KAG还是有很多可以改进的地方，比如实体去重（根据刘博的建议可以再提交一遍给大模型）、还有为什么会产生大量Other实体（在构造图谱的那部分会检测该实体是否存在于Schema中，如果没有，就自动生成Other实体）

### 2月18日

1、尝试将extractor的第一次抽取后的结果，再提交给大模型一次，让其剔除重复的实体（实验证明产生重复实体的原因不是实体抽取，而是关系抽取）

2、为KAG新增xlsx文档读取功能，以支持现实新闻抽取，同时只提取content和title，保证样本的简洁度

3、在xlsx_scanner.py文件中，添加数据清洗功能，剔除多余的特殊字符，新闻结构变得清晰后，大模型的拒答率反而上升了。

### 2月19日

1、根据项目需要，学习相关的社区发现算法：louvain算法、 leiden算法

2、使用KAG在Neo4j中建立图谱，然后再从中抽取出事件实体，做一个事件实体的聚类和关联。

3、和大家讨论接下来的工作任务，不断debug实验社区算法

### 2月20日

1、使用500条数据抽取事件实体，存储在neo4j中，大约抽取了300多事件实体。

2、撰写louvain的测试代码（比较费劲，一堆bug）

3、不断地debug测试代码，完成初版louvain源码

### 2月21日

1. 继续debug louvain代码，成功跑通第一次实验
2. 第一次实验结果表现很差，将事件实体聚类为独立的实体，根据实验情况提出两点可改进方向：
   1. 将与事件实体相关的所有关系和实体全部引入
   2. 查看500条新闻媒体之间有没有彼此强关联的事件实体
3. 改进代码，得到还不错的实验结果。计划后面将schema改为5W1H，并且考虑是否删除kag的chunk机制
