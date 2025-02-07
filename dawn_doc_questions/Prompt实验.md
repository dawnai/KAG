# Prompt实验

### 抽取样例：

抽取样例采用新闻的格式，主题属于时政内容，测试KAG的人物画像能力。

```
[
    {
        "title": "特朗普计划遣送无证移民到关塔那摩湾",
        "text": "美国总统唐纳德·特朗普（Donald Trump，川普）已下令在关塔那摩湾建设一个移民拘留设施，他表示该设施将可容纳多达30,000人。他称，位于古巴的美国海军基地这个设施，会与其具有高度安全性的军事监狱分开，将用来关押“威胁美国人民最严重的非法移民犯罪者。特朗普周三表示：我们要把他们送到关塔那摩。关塔那摩湾长期以来一直被用来关押移民，这个做法受到一些人权组织的批评。位于古巴的美国海军基地以关押2001年“9/11”袭击事件后被捕的嫌疑人而闻名。该基地设有一个军事拘留中心和法庭，专门用于关押在乔治·W·布什（George W. Bush，布希）总统领导下的“反恐战争”期间被拘留的外国人。这个设施由布什于2002年建立，目前关押着15名被拘留者，包括被指控为9/11事件主谋的哈立德·谢赫·穆罕默德（Khalid Sheikh Mohammed）。这个数字较高峰期的近800名囚犯大幅减少。包括奥巴马在内的几位民主党总统，都曾承诺关闭该设施，但最终未能实现。此外，该基地还有一个小型、独立的设施，数十年来都用于拘留移民。这个设施被称为关塔那摩移民行动中心，在不同的共和党和民主党政府中均有使用。它主要关押那些试图非法乘船抵达美国的人，大部分人来自海地和古巴。特朗普的边境沙皇汤姆·霍曼（Tom Homan）告诉记者：我们只是要扩大现有的移民中心，并补充说，该设施将会由移民及海关执法局负责运营。"
    }
]
```

### **模型：**

由于DeepSeek的火爆，导致服务器紧张，所以采用**Kimi**，因为是小样例抽取实验，所以千亿参数的大模型之间抽取差异很小，用什么都可以。

### Schema：

<img src="C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\image-20250207141711850.png" alt="image-20250207141711850" style="zoom:50%;" />

### 目标：

抽取最重要的保证信息的准确性，在此基础上尽可能拓展图谱的多样性和丰富性。



### **实验一：**

分别是ner.py std.py triple.py

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
                        {"entity": "特朗普宣布美国将再次退出《巴黎协定》", "category": "NewsArticale"}
                    ]
            }
        ],
        "input": "$input"
    }    
        """
       
```

```
    template_zh = """
{
    "instruction": "input字段包含用户提供的上下文。命名实体字段包含从上下文中提取的命名实体，这些可能是含义不明的缩写、别名或俚语。为了消除歧义，请尝试根据上下文和您自己的知识提供这些实体的官方名称。请注意，具有相同含义的实体只能有一个官方名称。请按照提供的示例中的输出字段格式，以单个JSONArray字符串形式回复，无需任何解释。",
    "example": {
        "input": "特朗普宣布美国将再次退出《巴黎协定》，来源：新华网 | 2025年01月21日 09:12:47。原标题：特朗普宣布美国将再次退出《巴黎协定》新华社华盛顿1月20日电（记者刘亚南 邓仙来）美国总统特朗普20日签署行政令，宣布美国将再次退出旨在应对气候变化的《巴黎协定》。2015年，联合国气候变化大会达成《巴黎协定》，成为全球应对气候变化的重要成果。2017年6月，时任美国总统特朗普宣布美国将退出《巴黎协定》。2020年11月4日，美国正式退出该协定。此举遭到美国国内和国际社会的广泛批评。2021年1月20日，拜登就任总统首日签署行政令，宣布美国将重新加入《巴黎协定》。同年2月19日，美国正式重新加入《巴黎协定》。",
        "named_entities": [
            {"name": "特朗普", "category": "Person"},
            {"name": "刘亚南", "category": "Person"},
            {"name": "邓仙来", "category": "Person"},
            {"name": "拜登", "category": "Person"},
        ],
        "output": [
            {"name": "特朗普", "category": "Person", "official_name": "唐纳德·特朗普"},
            {"name": "美国总统特朗普", "category": "Symptom", "official_name": "唐纳德·特朗普"},
            {"name": "拜登", "category": "Symptom", "official_name": "乔·拜登"},
        ]
    },
    "input": $input,
    "named_entities": $named_entities,
}    
    """
```

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
```

结果：

num_nodes": 50, "num_edges": 76

经过查看，发现以下信息：

![image-20250206170322807](C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\image-20250206170322807.png)

我觉得抽取了太多的schema的chunk文本块，有些是有一点冗余的。表现好的地方在于大模型自己丰富了官方名称。

### **实验二：**

改动ner.py的内容：

```
template_zh = """
    {
        "instruction": "你是命名实体识别的专家。请从输入中提取与模式定义匹配的实体。如果不存在该类型的实体，请返回一个空列表。请尽可能严格schema中的实体内容进行抽取，保持知识图谱的准确性和精简性。并且以JSON字符串格式回应。你可以参照example进行抽取。",
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
                        {"entity": "特朗普宣布美国将再次退出《巴黎协定》", "category": "NewsArticale"}
                    ]
            }
        ],
        "input": "$input"
    }    
        """
```

结果：

"num_nodes": 40, "num_edges": 60

经过查看，发现以下信息：

![graph (1)](C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\graph (1).svg)

可以看到，比上一次的图谱精简了一些，并且同样补充了官方名称，冗余现象大大减少，Chunk的文本块也减少了很多。但是仍有一部分是属于没有必要的文本，下一步需要修改关系提取了，或者将schema的文本块给删除。

### **实验三：**

有一些重复的实体提取，比如“移民及海关执法局”重复提取了两次，再次修改ner.py，让其保证相同实体只提取一次

结果：

"num_nodes": 56, "num_edges": 86

![graph (2)](C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\graph (2).svg)

这次效果我觉得还行，没有过多的冗余实体，又保证了图谱的多样性和丰富性。就是有一些实体没有任何关联，尝试修改关系抽取提取。

### **实验四：**

修改关系提取，让实体之间的关系更加准确和丰富。

结果：

"num_nodes": 25, "num_edges": 32

![graph (5)](C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\graph (5).svg)

可见，限制关系后，和Schema的内容保持一致了，就是人物、地点、新闻三个实体内容抽取。但是出现了很多重复的实体，比如“关塔那摩”

### 实验五：

又再次修改ner.py，限制实体抽取

结果：

"num_nodes": 43, "num_edges": 60

为什么又变多了？

![graph (6)](C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\graph (6).svg)

可以看到，几乎不存在相同的实体，但是文本块抽的比较多。

### 实验六：

修改schema，将实体内容限制在人物上。

目前schema如下：

<img src="C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\image-20250207143320861.png" alt="image-20250207143320861" style="zoom:67%;" />

抽取结果：（反复实验会导致API限速报错，需要一分钟后重试）

"num_nodes": 43, "num_edges": 62

![graph (7)](C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\graph (7).svg)

还是抽取了其他Other实体，明明已经修改了schema，并且提交了，为什么还会有？

### 实验七

新建一个项目，重新更新schema

KIMI疯狂限速！狗币东西（充了50块，解决访问问题）

然后实验结果:

"num_nodes": 58, "num_edges": 89

![graph (8)](C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\graph (8).svg)

出现大问题了！抽了一堆！虽然自定义schema删除了chunk、other字段，但是仍然抽取出来了，可见KAG会自动补充一些schema？

### 实验八：

对KIMI充了钱后，发现KIMI性能突然变了，抽取的东西更加多了，采用schema_constraint_extractor模式进行抽取

结果如下：

num_nodes": 1, "num_edges": 0,

![graph (10)](C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\graph (10).svg)

发现什么关系都没有，**这是正确的**，因为我的schema是表示人物撰写了什么、出版了什么、出席了什么活动。

保持相同prompt，然后改回schema_free_extractor结果：

"num_nodes": 31, "num_edges": 45

![graph (11)](C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\graph (11).svg)

这两个都出现相同的问题：人物实体存在问题，都将人物实体归结为了other实体。

---



后面更改了很多次，但是人物实体就是抽取不出来，人麻了！

更改schema：

```
namespace DawnMacSchema

Activity(活动会议):EntityType
    properties:
        desc(描述):Text
        nameEn(名称):Text

Location(地点):EntityType
    properties:
        desc(描述):Text
        nameEn(名字):Text

NewsArticale(新闻文章):EntityType
    properties:
        summary(摘要):Text
        author(作者):Text
        title(标题):Text
        publishTime(发布时间):Text

Person(人物): EntityType
     properties:
        info(信息): Text
            index: TextAndVector
        semanticType(语义类型): Text
            index: Text
        job(工作): Text
            constraint: MultiValue

Date(日期): EntityType
     properties:
        info(信息): Text
            index: TextAndVector
        semanticType(语义类型): Text
            index: Text

BaikeEvent(事件): EventType
     properties:
        subject(主体): Person
        participants(参与者): Person
            desc: the participants of event, such as subject and objects
            constraint: MultiValue
        time(时间): Date            
        location(地点): Location
        abstract(摘要): Text
            index: TextAndVector        
        semanticType(事件语义类型): Text
            desc: a more specific and clearly defined type, such as Professor or Actor for the Person type
            index: Text

Organization(组织机构): EntityType
     properties:
        info(信息): Text
            index: TextAndVector
        semanticType(语义类型): Text
            index: Text

Chunk(文本块): EntityType
     desc: A chunk refers to a segment of text.
     properties:
        content(内容): Text
          index: TextAndVector


```

然后抽取结果：

![graph (12)](C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\graph (12).svg)



人物还是没抽出来！

把ner改为默认试一试：

![graph (13)](C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\graph (13).svg)

总算是抽出来了，这说明是prompt问题！

明天继续实验！

**总结：**

1、单一样本抽取不合理，实验结果难以信服，后续增加抽取样例

2、模型不稳定，使用在线API方式实验会导致访问频繁，导致抽取失败（KIMI充钱可以解锁并发数），更改模型后，相同prompt表现差异巨大。后续建议使用OLLAMA本地部署模型

3、schema必须和prompt适配，否则抽取效果很差。



