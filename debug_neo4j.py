from obsidian_kg.config import Config
from obsidian_kg.web import _get_neo4j_driver

def debug_neo4j():
    config = Config.from_env()
    driver = _get_neo4j_driver(config)
    try:
        with driver.session() as session:
            labels = session.run("CALL db.labels()").value()
            rel_types = session.run("CALL db.relationshipTypes()").value()
            
            print(f"Labels: {labels}")
            print(f"Relationship Types: {rel_types}")
            
            # Check edge property keys
            edge_props = session.run("MATCH ()-[r:RELATES_TO]->() RETURN keys(r) LIMIT 1").data()
            print(f"RELATES_TO Edge Property Keys: {edge_props}")
            
            # Check a few sample edges
            samples = session.run("MATCH (a)-[r:RELATES_TO]->(b) RETURN properties(r) LIMIT 3").data()
            for i, s in enumerate(samples):
                print(f"Sample {i} properties: {s['properties(r)']}")

    finally:
        driver.close()

if __name__ == "__main__":
    debug_neo4j()
