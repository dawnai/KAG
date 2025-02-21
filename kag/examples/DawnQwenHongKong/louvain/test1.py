from neo4j import GraphDatabase
import pandas as pd
import networkx as nx
from community import community_louvain
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    def close(self):
        self.driver.close()

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

    def detect_communities(self):
        """执行Louvain社区发现"""
        logger.info("开始社区发现...")
        
        # 计算最佳分区
        self.partition = community_louvain.best_partition(self.G, weight='weight')
        
        # 将结果合并到节点数据
        self.nodes_df['community'] = self.nodes_df['node_id'].map(self.partition)
        
        logger.info(f"发现 {self.nodes_df['community'].nunique()} 个社区")

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

    def analyze_communities(self):
        """分析社区分布"""
        community_dist = self.nodes_df.groupby('community').size().sort_values(ascending=False)
        logger.info("社区分布统计:\n" + community_dist.to_string())
    

if __name__ == "__main__":
    # 配置信息
    config = {
        "uri": "bolt://172.20.158.20:7687",
        "user": "neo4j",
        "password": "neo4j@openspg",
        "db_name": "dawnqwenhongkong"
    }

    processor = Neo4jLouvainProcessor(**config)

    try:
        # Step 1: 数据导出
        processor.export_data()
        
        # 检查数据是否有效
        if processor.nodes_df.empty:
            raise ValueError("未找到任何节点，请检查标签名称")
        if processor.edges_df.empty:
            logger.warning("未找到节点之间的关系，社区发现可能不准确")

        # Step 2: 构建图
        processor.build_graph()
        
        # Step 3: 社区发现
        processor.detect_communities()
        
        # Step 4: 结果分析
        processor.analyze_communities()
        
        # Step 5: 写回结果
        processor.write_results()

    except Exception as e:
        logger.error(f"流程执行失败: {str(e)}")
    finally:
        processor.close()
        logger.info("处理完成，连接已关闭")
