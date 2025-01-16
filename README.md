# KAG源码拆解

<img src="https://dawnai.cloud/img/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20230206145629.ae57fb1d.jpg" alt="logo" style="zoom:50%;" />

用于dawn对KAG这个项目源码理解任务 :tada:

### 

### **1、基础环境**搭建

由于KAG是基于OpenSPG实现的，首先需要**搭建SPG服务**

```shell
Linux： curl -sSL https://raw.githubusercontent.com/OpenSPG/openspg/refs/heads/master/dev/release/docker-compose.yml -o docker-compose.yml docker compose -f docker-compose.yml up -d
```

### **2、conda环境**

这一步注意安装python10的虚拟环境就好，我在windows上的环境名称是kag_py10，然后拿去源项目用本地安装方法（pip install -e .）进行依赖包安装。

### **3、创建知识库**

进入项目examples目录

```
cd kag/examples
```

编辑项目配置

```
 ./example_config.yaml （linxu倒是可以用vim，windows就记事本就行）
```

改为如下：（**虽然我这里使用了中文注释，但是使用的时候一定要删除中文注释！否则会报编码错误！**）

```yaml
# 这个用于信息抽取
openie_llm: &openie_llm
  api_key: sk-5628e0dd09de41feb558b655ad409f0f
  base_url: https://api.deepseek.com
  model: deepseek-chat
  type: maas

# 用于生成回答
chat_llm: &chat_llm
  api_key: sk-5628e0dd09de41feb558b655ad409f0f
  base_url: https://api.deepseek.com
  model: deepseek-chat
  type: maas

# 字符转向量，使用硅基流动提供的bge-m3大模型，如果是在产品端，系统已经内置了bge-m3
vectorize_model: &vectorize_model
  api_key: sk-svefxzzoouzucawpdswgqkodaavgujuzpjcugwkqqwswiclx
  base_url: https://api.siliconflow.cn/v1/
  model: BAAI/bge-m3
  type: openai
  vector_dimensions: 1024
vectorizer: *vectorize_model
```

**创建项目**（与产品模式中的知识库一一对应）：

```
knext project create --config_path ./example_config.yaml
```

> [!WARNING]
>
> 报错：
>
> UnicodeDecodeError: 'gbk' codec can't decode byte 0xac in position 415: illegal multibyte sequence
>
> 经过大量debug后，发现是自己在yaml文件中的注释写了中文，就会报这种编码错误。
>
> 草！

**目录初始化**

创建项目之后会在kag/examples目录下创建一个与project配置中namespace字段同名的目录（示例中为TwoWikiTest），并完成KAG项目代码框架初始化。用户可以修改下述文件的一个或多个，完成业务自定义图谱构建&推理问答。

```
.
├── builder
│   ├── __init__.py
│   ├── data
│   │   └── __init__.py
│   ├── indexer.py
│   └── prompt
│       └── __init__.py
├── kag_config.yaml
├── reasoner
│   └── __init__.py
├── schema
│   ├── TwoWikiTest.schema
│   └── __init__.py
└── solver
    ├── __init__.py
    ├── data
    │   └── __init__.py
    └── prompt
        └── __init__.py
```

**导入文档**

进入项目：

```
cd kag/examples/TwoWikiTest
```

获取语料数据：

2wiki 数据集的测试语料数据为 kag/examples/2wiki/builder/data/2wiki_corpus.json，有 6119 篇文档，和 1000 个问答对。为了迅速跑通整个流程，目录下还有一个 2wiki_corpus_sub.json文件，只有 3 篇文档，我们以该小规模数据集为例进行试验。其复制到 TwoWikiTest 项目的同名目录下：（**或者使用自己的json文件，记住更改路径就行。**）

```
cp ../2wiki/builder/data/2wiki_sub_corpus.json builder/data/
```

编辑 schema（可选)

编辑 schema/TwoWikiTest.schema 文件，schema 文件格式参考 [知识建模](https://openspg.yuque.com/ndx6g9/0.6/fzhov4l2sst6bede) 相关章节。

提交 schema 到服务端

```
$ knext schema commit
```

执行构建任务

在builder/indexer.py文件中定义任务构建脚本：

```python
import os
import logging
from kag.common.registry import import_modules_from_path

from kag.builder.runner import BuilderChainRunner

logger = logging.getLogger(__name__)


def buildKB(file_path):
    from kag.common.conf import KAG_CONFIG

    runner = BuilderChainRunner.from_config(KAG_CONFIG.all_config["kag_builder_pipeline"])
    runner.invoke(file_path)

    logger.info(f"\n\nbuildKB successfully for {file_path}\n\n")


if __name__ == "__main__":
    import_modules_from_path(".")
    dir_path = os.path.dirname(__file__)
    # 将file_path设置为之前准备好的语料文件路径
    file_path = os.path.join(dir_path, "data/2wiki_sub_corpus.json")

    buildKB(file_path)
```

运行indexer.py脚本完成非结构化数据的图谱构建。

```
cd builder 

python indexer.py
```

完成后就可以去产品端自己看结果
