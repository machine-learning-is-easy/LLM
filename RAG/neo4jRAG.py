from neo4j import GraphDatabase

# Connect to the Neo4j database
uri = "bolt://localhost:7687"  # Adjust for your Neo4j instance
username = "neo4j"
password = "your-password"

driver = GraphDatabase.driver(uri, auth=(username, password))

# Define a session to interact with the database
def create_session():
    return driver.session()


def create_graph(session):
    # Create nodes and relationships
    query = """
    MERGE (p:Person {name: 'Alice'})
    MERGE (p)-[:HAS_SKILL]->(s:Skill {name: 'Data Science'})

    MERGE (p2:Person {name: 'Bob'})
    MERGE (p2)-[:HAS_SKILL]->(s2:Skill {name: 'Machine Learning'})

    MERGE (p)-[:WORKS_WITH]->(p2)
    """
    session.run(query)


# Use the session to create the graph
with create_session() as session:
    create_graph(session)

def retrieve_knowledge(session, query_text):
    # This query retrieves people and skills related to the input text
    cypher_query = """
    MATCH (p:Person)-[:HAS_SKILL]->(s:Skill)
    WHERE s.name CONTAINS $query_text
    RETURN p.name AS person, s.name AS skill
    """
    result = session.run(cypher_query, query_text=query_text)
    knowledge = []
    for record in result:
        knowledge.append(f"{record['person']} is skilled in {record['skill']}.")
    return " ".join(knowledge)


import openai


# Function to use RAG
def rag_inference(user_input):
    # Retrieve knowledge from Neo4j
    with create_session() as session:
        knowledge = retrieve_knowledge(session, user_input)

    # Combine knowledge and pass to the LLM
    prompt = f"Here is some information related to your query: {knowledge}\nNow, answer the question: {user_input}"

    response = openai.Completion.create(
        engine="text-davinci-003",  # Or any available LLM engine
        prompt=prompt,
        max_tokens=100
    )

    return response.choices[0].text.strip()


# Example usage
user_input = "Tell me about skills in data science."
response = rag_inference(user_input)
print("Response:", response)

