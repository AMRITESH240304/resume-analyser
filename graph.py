import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

skills = ["Machine Learning", "Deeplearning", "NLP", "LLM"]
languages = ["Python", "Javascript", "C++", "C"]

edges = [
    ("Machine Learning", "Python"),
    ("Deeplearning", "Python"),
    ("NLP", "Python"),
    ("LLM", "Python"),
    ("Machine Learning", "C++"),
    ("Deeplearning", "C++"),
    ("NLP", "C++"),
    ("LLM", "C++"),
    ("Machine Learning", "C"),
    ("Deeplearning", "C"),
    ("NLP", "C"),
    ("LLM", "C"),
    ("Machine Learning", "Javascript"),
    ("Deeplearning", "Javascript"),
    ("NLP", "Javascript"),
    ("LLM", "Javascript")
]

G.add_edges_from(edges)

pos = nx.spring_layout(G) 

nx.draw_networkx_nodes(G, pos, node_size=7000, node_color='skyblue')

nx.draw_networkx_edges(G, pos, edgelist=edges, arrows=True)

nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif')

edge_labels = {edge: '' for edge in edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

plt.title("Skills and Programming Languages Network Graph")

plt.savefig("skills_languages_network_graph.png")

print("Graph has been saved as 'skills_languages_network_graph.png'")
