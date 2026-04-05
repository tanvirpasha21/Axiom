# In Python
from axiom import LiteratureAgent

agent = LiteratureAgent(
    backend="ollama", model="mistral",
    paper_source="local+llm",       # your store + LLM knowledge
    db_path="axiom_papers.jsonl",   # your training file
)
results = agent.search("fraud detection")
print(results)