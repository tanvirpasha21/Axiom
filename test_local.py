# In Python
from axiom import LiteratureAgent

agent = LiteratureAgent(
    backend="openrouter",
    api_key="YOUR_OPENROUTER_API_KEY",  # set OPENROUTER_API_KEY env var instead
    model="openai/gpt-4o"
)
results = agent.search("fraud detection transformer models")