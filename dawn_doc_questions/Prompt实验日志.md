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

### **Schema：**

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

总算是把人物实体抽出来了，这说明是prompt问题！

明天继续实验！

**总结：**

1、单一样本抽取不合理，实验结果难以信服，后续增加抽取样例

2、模型不稳定，使用在线API方式实验会导致访问频繁，导致抽取失败（KIMI充钱可以解锁并发数），更改模型后，相同prompt表现差异巨大。后续建议使用OLLAMA本地部署模型

3、schema必须和prompt适配，否则抽取效果很差。



## 2月8号

昨天晚上开了会，有个针对香港的舆情监测项目

所以今天改变一下实验方向，将模型特换成Qwen2.5 72B，语言换成香港的特色繁体+中英混合语言，同时更改schema，替换成舆情监控的schema。

### 模型：

qwen2.5-72b-instruct，采用阿里云api

### 待抽取样本：

```
[
    {
	"title": "珍惜生命│牛頭角彩盈邨女子燒炭 友人上門聞煤氣味揭發",
    "text": "牛頭角彩盈邨盈富樓一單位，今（8日）中午12時58分，一名女子在單位內燒炭自殺，其友人訪上址在門外嗅到煤氣味，心感不妙，報警求助，消防到場出動一喉一煙帽隊，女事主昏迷被送往聯合醫院搶救。警方將事件列企圖自殺跟進，調查其尋死原因。自殺求助熱線：「情緒通」精神健康支援熱線：18111 香港撒瑪利亞防止自殺會： 2389 2222生命熱線： 2382 0000明愛向晴軒： 18288社會福利署： 2343 2255撒瑪利亞會熱線(多種語言)： 2896 0000東華三院芷若園： 18281醫管局精神康專線： 2466 7350賽馬會青少年情緒健康網上支援平台「Open 噏」：http://www.openup.hk"
    },
    {
    "title":"特朗普稱DeepSeek不會構成國安威脅 料美國最終「受益」",
    "text":"中國AI企業DeepSeek（深度求索）近日發佈的開源模型引起全球關注。綜合外媒報道，美國總統特朗普昨日（7日）對此表示，DeepSeek不會對國家安全構成威脅，而且美國最終可以「受益」。特朗普提到，DeepSeek是一項正在發展的技術，如果發展正確的話，美國將會從中受益，因為現在所涉及的人工智能成本將比人們最初想像的低很多，「這是一件好事」。他又強調，這是非常積極的發展，而非壞事。另一方面，韓國、意大利、澳洲、印度及日本等國先後傳出禁止或限制使用 DeepSeek。中國外交部發言人郭嘉昆周四對此表示，中國政府高度重視並依法保護數據隱私和安全，從來沒有也不會要求企業或個人以違法的形式採集或存儲數據。他又指，中方一貫反對泛化國家安全概念、將經貿科技問題政治化的做法，中方將堅定維護中國企業的合法權益。"
    },
    {
    "title":"啟鑽商場酒樓冒煙消防救熄 疑抽煙系統失靈",
    "text":"鑽石山彩虹道啟鑽苑啟鑽商場一樓，今（8日）中午12時47分，一間中式酒樓廚房冒煙，職員報案求助，消防接報到場，開喉救熄，事件中無人受傷，毋須疏散。經調查後，相信是油炸食物時抽煙系統失靈，並無可疑。"
    },
    {
    "title":"公屋擲物狂亂丟有料安全套 兼空降濕漉漉尿墊 苦主願贈電動XX解決｜Juicy叮",
    "text":"青衣公屋出現擲物狂，懷疑亂丟「針筒、用完安全套、洗頭水樽、用過嘅女士用品......寵物尿墊」，部分垃圾擱於下層住戶的晾衣架上，讓鄰居不勝其煩。居長康邨的苦主早前在facebook群組「青衣街坊吹水會」上載相片發帖訴苦，指「即係咁，（長康邨）康Ｘ樓Ｘ座20以上嘅缺德住戶，如果屋企冇垃圾桶嘅話，我可以送一個俾你，如果你行動不便嘅話，我都可以送張電動輪椅（中國製）俾你，等你攞出去掉垃圾桶都方便啲。#已經唔係第一次 #講得出嘅垃圾都有執過 #好想房署快啲裝閉路電視」。相中見到在開揚景觀映襯下，晾衣架上擱有一大塊泛著黃色污漬且仍然濕漉漉的尿墊，窗邊還有一疊疑似空降的報紙。"
    },
    {
    "title":"台「太陽花女王」劉喬安波士頓落網 匿美逾5年被台灣通緝",
    "text":"美國總統特朗普上任後，針對打擊非法移民。美國移民及海關執法局（ICE）1月23日在一次針對性執法行動中，逮捕了一名在台灣因貪污、非法侵佔、詐欺和販毒而被通緝的台灣通緝犯。而官方公布的照片顯示，她就是「太陽花女王」劉喬安。劉喬安2019年5月合法以臨時訪客身份進入美國，原本必須在2019年8月離境，但她並未遵守規定。台灣刑事局國際刑警科今年初掌握劉女的行蹤後，立即將情報通知美方，美國移民暨海關執法局、波士頓警方等單位，1月23日在波士頓一家旅館將劉女逮補，刑事局今（8）日證實，已與美方合作正在協調遣返事宜，新北地檢署說，待警方將她解送歸案後，將由當日值班內勤檢察官處理。"
    }
]
```

### Schema:

```
namespace DawnQwenHongKong

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

三个prompt：

```
template_zh = """
    {
        "instruction": "你是命名实体识别的专家。现在你需要从香港新闻媒体中提取与模式定义匹配的实体（以繁体字为主）。如果不存在该类型的实体，请返回一个空列表。请以JSON字符串格式回应。你可以参照example进行抽取",
        "schema": $schema,
        "example": [
            {
                "input": "深圳南山今日（8日）發生巴士撞上巴士站，事後釀成2死1傷。深圳公安通報事件。通報指出，2月8日10時許，在南山區沙河西路茶光村公交站，一公交車入站停靠時因司機突發疾病與站台發生碰撞，造成3名候車乘客受傷，其中2人經搶救無效死亡。經對公交車司機進行呼氣式酒精測試，結果為0mg/100ml。目前，事故正在進一步調查處理中。",
                "output": [
                            {
                                "name": "2月8日10時",
                                "type": "Date",
                                "category": "Date",
                                "description": "深圳南山發生巴士撞上巴士站时间"
                            },
                            {
                                "name": "深圳南山區沙河西路茶光村公交站",
                                "type": "Location",
                                "category": "Location",
                                "description": "事故发生地点"
                            },
                            {
                                "name": "深圳南山發生巴士撞上巴士站，事後釀成2死1傷",
                                "type": "NewsArticale",
                                "category": "NewsArticale",
                                "description": "2月8日10時 发生的事故新闻 "
                            },
                            {
                                "name": "深圳公安",
                                "type": "Organization",
                                "category": "Organization",
                                "description": "深圳市公安局，负责该事故的调查"
                            },
                            {
                                "name": "呼氣式酒精測試",
                                "type": "Other",
                                "category": "Other",
                                "description": "呼氣式酒精測試，一种测试驾驶员是否酒驾的测试工具，该事故的驾驶员测试結果為0mg/100ml，没有酒驾"
                            },
                            {
                                "name": "公交車司機",
                                "type": "Person",
                                "category": "Person",
                                "description": "该事故的公交车驾驶员，因为突發疾病與站台發生碰撞"
                            }
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
        "input": "深圳南山今日（8日）發生巴士撞上巴士站，事後釀成2死1傷。深圳公安通報事件。通報指出，2月8日10時許，在南山區沙河西路茶光村公交站，一公交車入站停靠時因司機突發疾病與站台發生碰撞，造成3名候車乘客受傷，其中2人經搶救無效死亡。經對公交車司機進行呼氣式酒精測試，結果為0mg/100ml。目前，事故正在進一步調查處理中。",
        "named_entities": [
            {"name": "深圳公安", "category": "Organization"},
            {"name": "呼氣式酒精測試", "category": "Other"}
        ],
        "output": [
            {"name": "深圳公安", "category": "Organization", "official_name": "深圳市公安局"},
            {"name": "呼氣式酒精測試", "category": "Other", "official_name": "体内酒精测定仪，breathalyzer"}
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
        "instruction": "您是一位专门从事开放信息提取（OpenIE）的专家。请从input字段的文本中提取可能的关系（包括主语、谓语、宾语），并按照JSON格式列出它们，须遵循example字段的示例格式。请注意以下要求：1. 每个三元组应至少包含entity_list实体列表中的一个，但最好是两个命名实体。2. 明确地将代词解析为特定名称，以保持清晰度。",
        "entity_list": $entity_list,
        "input": "$input",
        "example": {
            "input": "深圳南山今日（8日）發生巴士撞上巴士站，事後釀成2死1傷。深圳公安通報事件。通報指出，2月8日10時許，在南山區沙河西路茶光村公交站，一公交車入站停靠時因司機突發疾病與站台發生碰撞，造成3名候車乘客受傷，其中2人經搶救無效死亡。經對公交車司機進行呼氣式酒精測試，結果為0mg/100ml。目前，事故正在進一步調查處理中。",
            "entity_list": [
                {"name": "2月8日10時", "category": "Date"},
                {"name": "深圳南山區沙河西路茶光村公交站", "category": "Location"},
                {"name": "深圳南山發生巴士撞上巴士站，事後釀成2死1傷", "category": "NewsArticale"},
                {"name": "深圳公安", "category": "Organization"},
                {"name": "呼氣式酒精測試", "category": "Other"},
            ],
            "output":[
                ["深圳南山區沙河西路茶光村公交站", "发生", "巴士撞上巴士站事故"],
                ["深圳公安", "通报", "造成3名候車乘客受傷，其中2人經搶救無效死亡"],
                ["公交车司机", "进行", "呼氣式酒精測試"],
                ["呼氣式酒精測試", "结果为", "0mg/100ml"],
                ["2月8日10時", "发生", "深圳南山發生巴士撞上巴士站，事後釀成2死1傷"]
            ]
        }
    }    
        """
```

### 实验一：

保持上述prompt和schema以及模型，抽取结果：

```
"num_edges": 25, "num_subgraphs": 1
"num_nodes": 22, "num_edges": 37, "num_subgraphs": 1
"num_edges": 44, "num_subgraphs": 1
"num_nodes": 40, "num_edges": 59, "num_subgraphs": 1
```

![graph (15)](C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\graph (15)-1739000859151-3.svg)

结果发现5个样例只抽取了4个？同时时间被赋予了官方名称。

### 实验二：

最后一个案例没有被抽取，怀疑是我mac服务器刚刚熄屏的原因或者切片超出Qwen的maxtoken，保持参数不动，再试一下：

结果：

```
{"num_nodes": 21, "num_edges": 31, "num_subgraphs": 1}}}
{"num_nodes": 33, "num_edges": 46, "num_subgraphs": 1}}}
{"num_nodes": 48, "num_edges": 71, "num_subgraphs": 1}}}
{"num_nodes": 55, "num_edges": 88, "num_subgraphs": 1}}}
{"num_nodes": 33, "num_edges": 55, "num_subgraphs": 1}}}

```

![graph (16)](C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\graph (16).svg)

这次就行了，KAG默认分片长度是100000，Qwen 2.5 72B最多支持上下文131072，虽然没有超过模型的maxtoken，但是如果拉取一篇长篇报道，Qwen 2.5 72B可能就难以胜任。

实验发现，有些报道之间是有关联的，这些属于隐藏的信息被挖掘了出来。

后续可以进一步更改Schema，将抽取重点放在香港本土环境中，甚至限制在九龙地区，不必抽取香港之外的其他信息。

### 实验三：

更改schema：

针对人物新增所属机构，针对事件实体新增impactLevel舆情等级和sentimentScore事件的情感倾向打分

```
namespace DawnQwenHongKong

Activity(活动会议):EntityType
    properties:
        desc(描述):Text
        nameEn(名称):Text

Location(九龙地点):EntityType
    properties:
        desc(描述):Text
        nameEn(名字):Text

NewsArticale(新闻文章):EntityType
    properties:
        summary(摘要):Text
        author(作者):Text
        title(标题):Text
        publishTime(发布时间):Text
        source(来源):Text
        keywords(关键词):Text

Person(人物): EntityType
     properties:
        info(信息): Text
            index: TextAndVector
        semanticType(语义类型): Text
            index: Text
        job(工作): Text
            constraint: MultiValue
            organization(所属机构):Organization

Date(日期): EntityType
     properties:
        info(信息): Text
            index: TextAndVector
        semanticType(语义类型): Text
            index: Text
        

Organization(组织机构): EntityType
     properties:
        info(信息): Text
            index: TextAndVector
        semanticType(语义类型): Text
            index: Text

JiuLongEvent(事件): EventType
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
        impactLevel(影响级别):Text
        	desc: Quantify the impact level of events to facilitate public opinion analysis
        	index:Text
        sentimentScore(情感打分):Text
        	desc: Sentiment scoring of events
        	index:Text
        

Chunk(文本块): EntityType
     desc: A chunk refers to a segment of text.
     properties:
        content(内容): Text
          index: TextAndVector
```

同时针对prompt修改成只提取和香港相关的信息，将图谱精简一下

```
template_zh = """
    {
        "instruction": "你是命名实体识别的专家。现在你需要从香港新闻媒体中提取与模式定义匹配的实体（以繁体字为主）。请注意！只需要提取和香港相关的实体，和香港无关的实体不必提取！（会出现很多其他地区的新闻！）如果不存在该类型的实体，请返回一个空列表。请以JSON字符串格式回应。你可以参照example进行抽取",
        "schema": $schema,
        "example": [
            {
                "input": "啟鑽商場酒樓冒煙消防救熄 疑抽煙系統失靈，鑽石山彩虹道啟鑽苑啟鑽商場一樓，今（8日）中午12時47分，一間中式酒樓廚房冒煙，職員報案求助，消防接報到場，開喉救熄，事件中無人受傷，毋須疏散。經調查後，相信是油炸食物時抽煙系統失靈，並無可疑。",
                "output": [
                            {
                                "name": "今（8日）中午12時47分",
                                "type": "Date",
                                "category": "Date",
                                "description": "香港啟鑽商場酒樓冒烟时间"
                            },
                            {
                                "name": "鑽石山彩虹道啟鑽苑啟鑽商場",
                                "type": "Location",
                                "category": "Location",
                                "description": "冒煙事故发生地点"
                            },
                            {
                                "name": "啟鑽商場酒樓冒煙消防救熄 疑抽煙系統失靈",
                                "type": "JiuLongEvent",
                                "category": "JiuLongEvent",
                                "description": "今（8日）中午12時47分 发生的消防新闻 "
                            },
                            {
                                "name": "消防",
                                "type": "Organization",
                                "category": "Organization",
                                "description": "香港消防队，负责该事故的调查"
                            },
                            {
                                "name": "抽煙系統",
                                "type": "Other",
                                "category": "Other",
                                "description": "一种排除烟雾系统，由于该系统失灵导致冒烟情况产生"
                            },
                            {
                                "name": "職員",
                                "type": "Person",
                                "category": "Person",
                                "description": "商场工作人员報案求助"
                            }
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
```

```
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
```

结果：

```
"num_edges": 30, "num_subgraphs": 1}}}
"num_nodes": 25, "num_edges": 36, "num_subgraphs": 1}}}
"num_nodes": 17, "num_edges": 24, "num_subgraphs": 1}}}
"num_nodes": 26, "num_edges": 34, "num_subgraphs": 1}}}
"num_nodes": 31, "num_edges": 43, "num_subgraphs": 1}}}
```

![graph (17)](C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\graph (17).svg)

虽然Prompt中明确提示不要抽取其他地区的相关信息，但是大模型还是抽取了深圳、南山、特朗普……

最关键的JiuLongEvent实体是一个都没有抽出来……

大概率是prompt太简单的原因，需要找一个更具有代表性的例子，刘博今天刚刚发了评论爬取，后续会发香港的新闻爬取，明天的实验从那里面拿。

**总结：**

1、Qwen 2.5是完全可以胜任此次任务，72B的参数恰到好处，不仅可以本地部署还可以防止提取过多实体（千亿大模型提取太多无关实体）

2、人物信息提取是个难点，大模型总是将人物归结为other，但是通过昨天的实验，我明白是prompt示例中人物样例太少的原因

3、JiuLongEvent实体提取也面临和人物一样的问题

4、schema_free_extractor相比起Schema_Constraint_Extractor好用，主要原因是目前schema还不够完善，需要大模型发挥自身的知识库进行补充

5、明天按照已知问题进行改进。



## 2月10号

抽取刘博给出的实际新闻样例，因为是xlsx文件，所以不能直接进行抽取，需要转换成其他文件。

最开始转化成txt，发现效果不好，最终换成csv文件，但是由于体量太大，单个文件一瞬间抽取1k多实体，所以需要对样例进行删改。

同时昨天大部分时间花在dify部署问题上，所以只进行了一次实验。

## 2月11号

### 实验一

吸取昨天的经验，将待抽取样例删改至10条新闻：

![image-20250211150724748](C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\image-20250211150724748.png)

转化为csv后，试着抽取一下，然后报错：

```
Done process 10 records, with 0 successfully processed and 10 failures encountered.
The log file is located at ckpt\kag_checkpoint_0_1.ckpt. Please access this file to obtain detailed task statistics.
```

没有一条成功？

主要报错如下：

```
TypeError: kag.builder.model.chunk.Chunk() got multiple values for keyword argument 'id
```

删除kag_config.yaml文件中reader下面的content_col: text   id_col: idx    name_col: title 参数。

---

然后再执行一遍：

结果：

```
Done process 10 records, with 9 successfully processed and 1 failures encountered.
The log file is located at ckpt\kag_checkpoint_0_1.ckpt. Please access this file to obtain detailed task statistics.
```

成功了9个，然后有一条抽取失败，查看抽取失败的信息：

```
openai.BadRequestError: Error code: 400 - {'error': {'code': 'data_inspection_failed', 'param': None, 'message': 'Input data may contain inappropriate content.', 'type': 'data_inspection_failed'}}
```

看来是Qwen 2.5 检测到输入数据违反了相关规定，拒绝给出答复，新闻中可能具备暴力、血腥元素。

再看一下抽取效果：

```shell
{"num_nodes": 25, "num_edges": 36, "num_subgraphs": 1}}}
{"num_nodes": 45, "num_edges": 66, "num_subgraphs": 1}}}
{"num_nodes": 27, "num_edges": 40, "num_subgraphs": 1}}}
{"num_nodes": 19, "num_edges": 28, "num_subgraphs": 1}}}
{"num_nodes": 33, "num_edges": 49, "num_subgraphs": 1}}}
{"num_nodes": 21, "num_edges": 31, "num_subgraphs": 1}}}
{"num_nodes": 34, "num_edges": 52, "num_subgraphs": 1}}}
{"num_nodes": 67, "num_edges": 100, "num_subgraphs": 1}}}
{"num_nodes": 33, "num_edges": 49, "num_subgraphs": 1}}}
```

![graph (18)](C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\graph (18)-1739260556905-2.svg)

问题也是很明显，一眼看过去全是粉红色实体，这意味着大模型只抽取了Others实体，地名、任务、事件实体那是一个都没有抽取。当然，这也和新闻材料相关，我仔细看了那10个例子，是存在大量人名、地名、还有事件的。

推测可能是Qwen 2.5 72B不太够用，切换成千亿大模型KIMI试试看

### 实验二：

切换成千亿大模型Kimi后，结果如下：

```
Done process 10 records, with 7 successfully processed and 3 failures encountered.
```

成功了7条，失败3条

主要错误如下：

```
openai.BadRequestError: Error code: 400 - {'error': {'code': 400, 'message': 'The request was rejected because it was considered high risk', 'param': 'prompt', 'type': 'content_filter', 'innererror': {}}}

json.decoder.JSONDecodeError: Expecting value: line 4 column 13 (char 40)
json.decoder.JSONDecodeError: Unterminated string starting at: line 2 column 14 (char 15)
```

一个是Kimi大模型检测到有害内容拒绝回答，另一个是回答的JSON格式有误。由此可见，Kimi的安全审查机制要避Qwen 2.5 更加严格

查看抽取结果：

```
{"num_nodes": 19, "num_edges": 27, "num_subgraphs": 1}}}
{"num_nodes": 33, "num_edges": 49, "num_subgraphs": 1}}}
{"num_nodes": 25, "num_edges": 37, "num_subgraphs": 1}}}
{"num_nodes": 33, "num_edges": 49, "num_subgraphs": 1}}}
{"num_nodes": 35, "num_edges": 52, "num_subgraphs": 1}}}
{"num_nodes": 37, "num_edges": 55, "num_subgraphs": 1}}}
{"num_nodes": 52, "num_edges": 79, "num_subgraphs": 1}}}

```

![graph (19)](C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\graph (19).svg)

同样的问题，几乎所有的实体全部被抽取为Others，这次证明不是Qwen2.5的事，而是prompt和schema

### 实验三：

重写prompt：

```
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
```

然后将模型切换成Qwen 2.5 72B

结果如下：

![graph (20)](C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\graph (20).svg)

完全没有用！奇怪了，不应该啊？不科学啊？

### 实验四：

仔细检查了一下终端的输出，发现我prompt好像重复了？今天为了方便起见，我存了一份繁体新闻的prompt，并且在同一文件夹下，只是改了文件名而已。

将繁体prompt文件转移到其他文件夹后，再进行一遍抽取:

![graph (22)](C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\graph (22).svg)

开始有了人物、事件、地点实体了！

事件实体如下：

<img src="C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\graph (23).svg" alt="graph (23)" style="zoom:67%;" />

但是有两条样例，仍然抽取了很多Others？其中一条属于养生的科普新闻，另外一条是英文的新闻：美国比特币矿工正在积累大量加密货币储备，以帮助他们抵御日益激烈的资源竞争所带来的利润空间不断压缩。

科普的那个被抽取Other可以理解，因为很多名词在schema中并没有提到，但是关于美国的这个新闻就不太应该了。

并且还有好几个样例都抽取失败了。

### 实验五：

![graph (24)](C:\Users\dawna\Desktop\KAG\dawn_doc_questions\assets\graph (24).svg)

对prompt再次微调一下，结果任然如此。

总结：

1. 大模型每次抽取结果具有一定的随机性
2. 对于某些含有暴力血腥类词汇的新闻媒体，大模型会拒绝回答
3. 大模型返回的JSON文件有时候会不符合要求
4. csv文档不一定是最好的xlsx的转换格式，后续还可以试一试其他格式
