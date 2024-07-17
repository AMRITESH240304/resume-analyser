from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Neo4j credentials from environment variables
URI = os.getenv('NEO4JDB_URI')
AUTH = (os.getenv('NEO4JDB_USERNAME'), os.getenv('NEO4JDB_PASSWORD'))

# Function to create nodes and relationships
def create_nodes_and_relationships(driver):
    with driver.session() as session:
        session.write_transaction(add_skills_projects_and_tech)

# Transaction function to create nodes and relationships
def add_skills_projects_and_tech(tx):
    skills = ["Machine Learning", "Deep Learning", "NLP", "LLM"]
    projects = [
        {
            "name": "Jai Kisan",
            "tech": ["Random Forest", "Sentimental Analysis", "Deep Learning"]
        },
        {
            "name": "Samvidhan Ai",
            "tech": ["Bart LLM model", "Qlora", "FastApi"]
        },
        {
            "name": "Stop Sign Generator",
            "tech": ["DCGAN", "FastApi", "YOLO", "SSD"]
        }
    ]
    development_techs = ["ML Algorithms", "TensorFlow", "Pytorch", "ResNet50", "VGG19", "EfficientNetB0", "CNN", "RNN", "FastApi", "NodeJs", "MongoDB"]

    # Create Skill nodes
    for skill in skills:
        tx.run("MERGE (s:Skill {name: $skill})", skill=skill)

    # Create Project and Tech nodes, and relationships
    for project in projects:
        tx.run("MERGE (p:Project {name: $project})", project=project["name"])
        for tech in project["tech"]:
            tx.run("MERGE (t:Technology {name: $tech})", tech=tech)
            tx.run(
                "MATCH (p:Project {name: $project}), (t:Technology {name: $tech}) "
                "MERGE (p)-[:USES]->(t)", 
                project=project["name"], tech=tech
            )

    # Create Development Technology nodes
    for dev_tech in development_techs:
        tx.run("MERGE (d:DevelopmentTech {name: $dev_tech})", dev_tech=dev_tech)

    # Create relationships between skills and technologies used in projects
    skill_tech_relationships = {
        "Machine Learning": ["Random Forest", "Sentimental Analysis", "DCGAN"],
        "Deep Learning": ["Deep Learning", "DCGAN"],
        "NLP": ["Bart LLM model"],
        "LLM": ["Bart LLM model", "Qlora"]
    }
    
    for skill, techs in skill_tech_relationships.items():
        for tech in techs:
            tx.run(
                "MATCH (s:Skill {name: $skill}), (t:Technology {name: $tech}) "
                "MERGE (s)-[:RELATED_TO]->(t)",
                skill=skill, tech=tech
            )

# Main code
with GraphDatabase.driver(URI, auth=AUTH) as driver:
    driver.verify_connectivity()
    print("Connected to Neo4j")

    create_nodes_and_relationships(driver)
    print("Created nodes and relationships in Neo4j")
