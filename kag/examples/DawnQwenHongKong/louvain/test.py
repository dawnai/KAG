from neo4j import GraphDatabase
import logging

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

    def close(self):
        self.driver.close()

    def run_cypher(self, cypher, parameters=None):
        with self.driver.session(database=self.db_name) as session:
            try:
                result = session.run(cypher, parameters)
                return [record.data() for record in result]
            except Exception as e:
                logger.error(f"Query Failed: {cypher}\nError: {str(e)}")
                raise

    def create_subgraph_projection(self):
        """创建精确的投影（针对JiuLongEvent及其关联节点）"""
        cypher = """
        CALL gds.graph.project.cypher(
            'jiulong_community',  // 投影名称
            // 节点查询：包含所有JiuLongEvent节点及其直接关联的Entity节点
            'MATCH (n) 
             WHERE "DawnQwenRemove.JiuLongEvent" IN labels(n) OR "Entity" IN labels(n)
             RETURN id(n) AS id, labels(n) AS labels',
            // 关系查询：JiuLongEvent与其他Entity节点之间的关系
            'MATCH (a)-[r]->(b)
             WHERE ("DawnQwenRemove.JiuLongEvent" IN labels(a) AND "Entity" IN labels(b))
                OR ("DawnQwenRemove.JiuLongEvent" IN labels(b) AND "Entity" IN labels(a))
             RETURN id(a) AS source, id(b) AS target, type(r) AS type'
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN *
        """
        return self.run_cypher(cypher)

    def run_louvain_algorithm(self):
        """执行Louvain算法（仅针对JiuLongEvent节点）"""
        cypher = """
        CALL gds.louvain.write('jiulong_community', {
            nodeLabels: ['DawnQwenRemove.JiuLongEvent'],  // 仅处理目标节点
            writeProperty: 'community'
        })
        YIELD communityCount, modularity
        RETURN *
        """
        return self.run_cypher(cypher)

    def verify_results(self):
        """验证JiuLongEvent节点的社区分配"""
        cypher = """
        MATCH (n:DawnQwenRemove.JiuLongEvent)
        WHERE n.community IS NOT NULL
        RETURN n.name AS name, n.community AS community
        LIMIT 10
        """
        return self.run_cypher(cypher)

    def cleanup_projection(self):
        """安全清理投影"""
        check_cypher = "CALL gds.graph.exists('jiulong_community') YIELD exists"
        try:
            exists = self.run_cypher(check_cypher)[0].get('exists', False)
            if exists:
                self.run_cypher("CALL gds.graph.drop('jiulong_community')")
                logger.info("投影已清理")
            else:
                logger.warning("投影不存在，跳过清理")
        except Exception as e:
            logger.error(f"清理失败: {str(e)}")

if __name__ == "__main__":
    NEO4J_URI = "bolt://172.20.250.18:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "neo4j@openspg"
    NEO4J_DB = "dawnqwenremove"

    processor = Neo4jLouvainProcessor(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DB)

    try:
        # 步骤1：创建精确投影
        logger.info("正在创建子图投影...")
        projection_result = processor.create_subgraph_projection()
        logger.info(f"投影创建成功: {projection_result}")

        # 步骤2：执行Louvain算法
        logger.info("正在执行Louvain算法...")
        louvain_result = processor.run_louvain_algorithm()
        logger.info(f"社区划分完成: {louvain_result}")

        # 步骤3：验证结果
        logger.info("验证前10条结果:")
        results = processor.verify_results()
        for record in results:
            print(f"节点: {record['name']} => 社区: {record['community']}")

    except Exception as e:
        logger.error(f"主流程失败: {str(e)}")
    finally:
        processor.cleanup_projection()
        processor.close()
