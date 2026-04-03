"""
Example: Using AXIOM with different LLM backends
"""

from axiom import LiteratureAgent

print("=" * 60)
print("AXIOM Local LLM Examples")
print("=" * 60)

# Example 1: Use local Llama via Ollama
print("\n1. Using Ollama with Llama2:")
print("-" * 40)
print("""
from axiom import LiteratureAgent

agent = LiteratureAgent(backend="ollama", model="llama2")
result = agent.search("fraud detection transformers")
print(result.summary)
""")

# Example 2: Use local Mistral (faster)
print("\n2. Using Ollama with Mistral (recommended):")
print("-" * 40)
print("""
agent = LiteratureAgent(backend="ollama", model="mistral")
result = agent.search("graph neural networks finance")
print(result.summary)
""")

# Example 3: Use Anthropic Claude (cloud)
print("\n3. Using Anthropic Claude (default):")
print("-" * 40)
print("""
agent = LiteratureAgent()  # Uses Claude, requires ANTHROPIC_API_KEY
result = agent.search("quantum computing applications")
print(result.summary)
""")

# Example 4: Use OpenAI GPT-4
print("\n4. Using OpenAI GPT-4:")
print("-" * 40)
print("""
agent = LiteratureAgent(backend="openai", model="gpt-4")
result = agent.search("deep learning medical imaging")
print(result.summary)
""")

# Example 5: CLI usage
print("\n5. Via CLI:")
print("-" * 40)
print("""
# With local Llama
axiom search "fraud detection" --backend ollama --model mistral

# With Claude (default)
axiom search "fraud detection"

# Set environment defaults
export AXIOM_BACKEND=ollama
export AXIOM_MODEL=mistral
axiom search "any query"  # Now uses local Mistral
""")

# Setup instructions
print("\n" + "=" * 60)
print("Setup Instructions")
print("=" * 60)

print("""
1. Install Ollama from https://ollama.ai

2. Pull a model:
   ollama pull llama2      # 4GB
   ollama pull mistral     # 5GB (recommended)
   ollama pull neural-chat # 5GB

3. Start Ollama server:
   ollama serve

4. Use in AXIOM:
   agent = LiteratureAgent(backend="ollama", model="mistral")
   result = agent.search("your research query")
   print(result.summary)

For cloud backends, set environment:
   export ANTHROPIC_API_KEY=sk-ant-...
   export OPENAI_API_KEY=sk-...
""")

print("\n" + "=" * 60)
print("Benchmark (response time for sample query)")
print("=" * 60)
print("""
Model            | Speed    | Quality    | Local | Cost
----------       | ------   | --------   | ----- | --------
Mistral (7B)     | Very Fast| Very Good  | Yes   | Free
Llama2 (7B)      | Fast     | Good       | Yes   | Free
Neural-Chat (7B) | Fast     | Good       | Yes   | Free
Claude Opus      | Medium   | Excellent  | No    | $$
GPT-4            | Slow     | Excellent  | No    | $$$
""")
