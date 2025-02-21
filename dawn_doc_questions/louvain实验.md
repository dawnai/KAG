# louvain实验（事件聚类）

### 目的：验证社区发现算法louvain在事件聚类上的可行性

### 步骤大纲：

1. 利用KAG先建立一个较大的知识图谱，包含大量的事件实体（从刘博提供的新闻数据中抽取500条左右）（前期的实验已经可以让KAG抽取大量事件实体了）
2. 利用neo4j提取出事件实体和相关的关系
3. 用louvain进行事件实体的聚类、事件关联

---

## 实验一

### 1、数据准备

首先使用500条新闻数据进行抽取，只提取content和title内容，并且进行数据清洗。

2月18号，我改进了kag，以支持xlsx文档提取和数据清洗，根据18号实验结果，数据清洗后，大模型API拒答率会提升，所以前期我多准备一些数据进行提取，以丰富图谱。

> [!WARNING]
>
> 报错：
> TypeError: object supporting the buffer API required
>
> 原因找了一下，是因为xlsx文件中content列或title列有空值，自己代码写的差了点
>
> 解决：给源码添加空值判断

**提取结果**：（只展示事件实体）

![graph (34)](./assets/graph (34).svg)

一共抽取了**12176**个实体节点，**21769**条关系，其中有361个事件实体，也就是有139条左右的新闻大模型拒绝回答，这和预期的情况差不多（我使用阿里云API，如果暨大采用离线Qwen2.5 72B，情况会好一些，拒答率没有那么高 ）。

目前几乎所有的事件实体都没有关联，都是独立的（抛开KAG的Chunk机制，因为KAG会链接同一个文本块产生的事件实体）。

### 2、开始执行louvain算法：

Neo4j的图数据科学库（Graph Data Science Library，简称GDS）提供了丰富的图算法支持，包括Louvain算法。

> [!WARNING]
>
> 报错：
>
> Error: {code: Neo.ClientError.Procedure.ProcedureCallFailed} {message: Failed to invoke procedure `gds.graph.project.cypher`: Caused by: java.lang.LinkageError: GDS 2.12.0 is not compatible with Neo4j version: 5.25.1}

GDS版本和neo4j版本不适配，这个就麻烦了，因为我neo4j是通过KAG的docker一起部署的，要改GDS需要去docker容器里面改，但是这个过程可能会删除我已经建立好的知识图谱。

改变思路，用python工具从neo4j中将图数据读取出来，然后使用python_louvain算法进行社区发现，为每个事件实体进行社区划分，然后为社区id相同的实体添加聚类实体。

#### **初版代码：**

导入必要的库

```python
from neo4j import GraphDatabase
import pandas as pd
import networkx as nx #构建图结构
from community import community_louvain#louvain算法
import logging
from tqdm import tqdm
```

定义流程类：

```python
class Neo4jLouvainProcessor:
    def __init__(self, uri, user, password, db_name):
            self.driver = GraphDatabase.driver(
                uri, 
                auth=(user, password),
                max_connection_lifetime=30,
                keep_alive=True
            )
            self.db_name = db_name
            self.nodes_df = None
            self.edges_df = None
```

从neo4j中导出数据：

```python
 def export_data(self):
        """从Neo4j导出节点和关系数据"""
        logger.info("开始从Neo4j导出数据...")
        
        # 导出节点
        node_query = """
        MATCH (n:`DawnQwenRemove.JiuLongEvent`) //这个 `` 有点搞心态啊，debug半天
        RETURN id(n) AS node_id, n.name AS name
        """
        with self.driver.session(database=self.db_name) as session:
            result = session.run(node_query)
            self.nodes_df = pd.DataFrame([dict(record) for record in result])
        
        logger.info(f"共导出 {len(self.nodes_df)} 个节点")
        
        # 导出关系
        rel_query = """
        MATCH (a:`DawnQwenRemove.JiuLongEvent`)-[r]-(b:`DawnQwenRemove.JiuLongEvent`)
        RETURN id(a) AS source, id(b) AS target, type(r) AS relationship_type
        """
        with self.driver.session(database=self.db_name) as session:
            result = session.run(rel_query)
            self.edges_df = pd.DataFrame([dict(record) for record in result])
        
        logger.info(f"共导出 {len(self.edges_df)} 条关系")
```

构建图结构：

```python
def build_graph(self):
        """构建NetworkX图"""
        logger.info("正在构建图结构...")
        
        # 创建无向图
        self.G = nx.Graph()
        
        # 添加节点
        self.G.add_nodes_from(self.nodes_df['node_id'].tolist())
        
        # 添加带权重的边（这里假设关系类型作为权重）
        edge_weights = self.edges_df.groupby(['source', 'target']).size().reset_index(name='weight')
        for _, row in tqdm(edge_weights.iterrows(), total=len(edge_weights), desc="添加边"):
            self.G.add_edge(row['source'], row['target'], weight=row['weight'])
        
        logger.info(f"图构建完成 | 节点数: {self.G.number_of_nodes()} | 边数: {self.G.number_of_edges()}")
```

社区发现算法：

```python
def detect_communities(self):
        """执行Louvain社区发现"""
        logger.info("开始社区发现...")
        
        # 计算最佳分区
        self.partition = community_louvain.best_partition(self.G, weight='weight')#使用louvain
        
        # 将结果合并到节点数据
        self.nodes_df['community'] = self.nodes_df['node_id'].map(self.partition)
        
        logger.info(f"发现 {self.nodes_df['community'].nunique()} 个社区")
```

将结果写回neo4j

```python
def write_results(self, batch_size=1000):
        """将社区结果写回Neo4j，并为每个社区创建聚类实体"""
        logger.info("开始将社区信息写回Neo4j...")
        
        # 准备数据
        data = self.nodes_df[['node_id', 'community']].to_dict('records')
        
        # 分批次写入社区信息
        for i in tqdm(range(0, len(data), batch_size), desc="写入社区信息进度"):
            batch = data[i:i+batch_size]
            
            cypher = """
            UNWIND $batch AS row
            MATCH (n) WHERE id(n) = row.node_id
            SET n.community = row.community
            """
            try:
                with self.driver.session(database=self.db_name) as session:
                    session.run(cypher, {'batch': batch})
            except Exception as e:
                logger.error(f"批次 {i//batch_size} 写入失败: {str(e)}")
                continue
        
        logger.info("社区信息已成功写回数据库")
        
        # 创建聚类实体并链接到相同社区的事件实体
        logger.info("开始创建聚类实体并链接到相同社区的事件实体...")
        
        # 获取所有社区
        communities = self.nodes_df['community'].unique()
        
        for community in tqdm(communities, desc="创建聚类实体进度"):
            # 创建聚类实体
            create_cluster_cypher = """
            CREATE (c:Cluster {community_id: $community_id})
            RETURN id(c) AS cluster_id
            """
            try:
                with self.driver.session(database=self.db_name) as session:
                    result = session.run(create_cluster_cypher, {'community_id': community})
                    cluster_id = result.single()['cluster_id']
            except Exception as e:
                logger.error(f"创建聚类实体失败: {str(e)}")
                continue
            
            # 获取该社区的所有事件实体
            community_nodes = self.nodes_df[self.nodes_df['community'] == community]['node_id'].tolist()
            
            # 分批次链接事件实体到聚类实体
            for i in range(0, len(community_nodes), batch_size):
                batch_nodes = community_nodes[i:i+batch_size]
                
                link_cypher = """
                UNWIND $batch_nodes AS node_id
                MATCH (n) WHERE id(n) = node_id
                MATCH (c) WHERE id(c) = $cluster_id
                CREATE (n)-[:BELONGS_TO]->(c)
                """
                try:
                    with self.driver.session(database=self.db_name) as session:
                        session.run(link_cypher, {'batch_nodes': batch_nodes, 'cluster_id': cluster_id})
                except Exception as e:
                    logger.error(f"链接事件实体到聚类实体失败: {str(e)}")
                    continue
        
        logger.info("聚类实体已成功创建并链接到事件实体")
```

#### 测试一下社区发现效果：

```cypher
MATCH (c:Cluster)-[r]-(n)
RETURN c, type(r) AS relationship_type, n
```

![graph (36)](./assets/graph (36).svg)





少数事件属于一个社区：

![graph (37)](./assets/graph (37).svg)

其中:green_heart: **绿色节点**代表聚类事件实体，:heartpulse: **粉红色**代表事件实体。

我查了一下neo4j数据库，一共建立了340个社区，数量接近事件总数。基本上事件实体都被归纳成了独立事件。

当然，出现这样的原因有很多，以下是一些可能原因：

1. 我初版源码在进行社区发现的时候，只引入了事件实体之间的关系，没有引入其他额外实体关系
2. 抽取的500条新闻中，稀疏性比较大，彼此之间并没有强关联

### 改进源码：

针对第一个问题，

这次引入事件实体以及相关的所有边（但是也只有Chunk）。

```python
	def export_data(self):
        """从Neo4j导出节点和关系数据"""
        logger.info("开始从Neo4j导出数据...")
        
        # 导出节点
        node_query = """
        MATCH (n:`DawnQwenHongKong.JiuLongEvent`)
        RETURN id(n) AS node_id, n.name AS name, labels(n) AS labels
        """
        with self.driver.session(database=self.db_name) as session:
            result = session.run(node_query)
            self.nodes_df = pd.DataFrame([dict(record) for record in result])
        
        logger.info(f"共导出 {len(self.nodes_df)} 个节点")
        
        # 导出关系(所有与事件实体相关的节点和边)
        rel_query = """
        MATCH (a:`DawnQwenHongKong.JiuLongEvent`)-[r]-(b)
        RETURN id(a) AS source, labels(a) AS source_labels,
            id(b) AS target, labels(b) AS target_labels,
            type(r) AS relationship_type
        """
        
        with self.driver.session(database=self.db_name) as session:
            result = session.run(rel_query)
            self.edges_df = pd.DataFrame([dict(record) for record in result])
        
        logger.info(f"共导出 {len(self.edges_df)} 条关系")
    def build_graph(self):
        """构建NetworkX图"""
        logger.info("正在构建图结构...")
        # 创建无向图
        self.G = nx.Graph()
        # 添加节点（包括所有相关节点）
        all_nodes = set(self.nodes_df['node_id'].tolist()).union(set(self.edges_df['source']).union(set(self.edges_df['target'])))
        self.G.add_nodes_from(all_nodes)
        # 添加带权重的边（这里假设关系类型作为权重）
        edge_weights = self.edges_df.groupby(['source', 'target']).size().reset_index(name='weight')
        for _, row in tqdm(edge_weights.iterrows(), total=len(edge_weights), desc="添加边"):
            self.G.add_edge(row['source'], row['target'], weight=row['weight'])

        logger.info(f"图构建完成 | 节点数: {self.G.number_of_nodes()} | 边数: {self.G.number_of_edges()}")
```

针对第二个问题，新闻之间是不是强相关？

查看一下JiuLongEvent事件实体之间的情况：

```cypher
MATCH (a:`DawnQwenHongKong.JiuLongEvent`)-[r]-(b)
RETURN a,r,b
```

![graph (39)](./assets/graph (39).svg)

结果显示，有些事件实体之间确实是有联系的，但是这联系也不是那么紧密，基本来自KAG的chunk机制。少数事件实体之间存在因果关系。

删除原本的聚类事件实体Cluster，然后再进行一次实验：

结果如下：

![graph (40)](./assets/graph (40).svg)

试一试只返回聚类事件实体和事件实体，删除chunk

![graph (41)](./assets/graph (41).svg)

这次效果很不错，社区数量大大减少，一共只建立了74个社区，再也不像第一次实验那样，给每个事件都创建一个聚类。

并且相同事件实体也都聚类到一个聚类实体上了！

## 实验二

虽然实验一取得了不错的效果，但是实验一仍然不足以说明louvain的可行性，因为返回的JiuLongEvent事件实体关系边是Chunk，而kag会自动连接同一个Chunk产生的所有JiuLongEvent事件实体，这变相为JiuLongEvent事件实体之间添加了强联系。如果不返回Chunk实体呢？只返回与事件相关的时间、地点、人物实体呢？毕竟项目要采用5W1H的形式。

问题来了，目前我的kag抽取并不是5w1h，schema还是我自定义的事件实体，所以抽取的JiuLongEvent事件实体几乎都没有和时间、地点、人物联系起来，都是通过Chunk连接，如下图所示：

![graph (42)](./assets/graph (42).svg)

下一步就是改schema，改成5W1H抽取，看情况决定要不要删除kag源码中assemble_sub_graph_with_chunk（）函数，让chunk不再链接到实体节点。





































