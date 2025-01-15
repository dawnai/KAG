# KAG相关问题



### 1、表示模型和生成模型有什么区别？

感觉是指代openie的信息抽取模型和





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



