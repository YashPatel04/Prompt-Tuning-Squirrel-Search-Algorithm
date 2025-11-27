from src.evaluation.llm_interface import OllamaInterface
from src.evaluation.fitness import FitnessCalculator
from src.evaluation.evaluator import Evaluator
from src.evaluation.cache import PromptCache
from src.genome.decoder import GenomeDecoder
import pandas as pd

# Load dataset
df = pd.read_csv('data/processed/dev.csv')
dataset = [
    {'text': row['text'], 'label': row['label']}
    for _, row in df.iterrows()
]

# Initialize Ollama (make sure it's running!)
# Run in terminal: ollama serve
# Then in another terminal: ollama pull mistral
llm = OllamaInterface(
    base_url="http://localhost:1561",
    model="gemma3:270m",
    temperature=0.0
)

# Initialize fitness calculator
fitness_calc = FitnessCalculator(task='sentiment_classification')

# Initialize cache
cache = PromptCache(cache_dir='data/cache')

# Initialize evaluator
evaluator = Evaluator(
    llm_interface=llm,
    fitness_calculator=fitness_calc,
    cache=cache,
    use_cache=True
)

# Test a prompt
test_prompt = """You are an expert sentiment analyst.

Classify the sentiment of this text as positive, negative, or neutral.

Think step by step before answering.

Text: {input}
Result:"""

print("Evaluating test prompt...")
fitness = evaluator.evaluate_prompt(test_prompt, dataset, sample_size=10)
print(f"Fitness score: {fitness}")

# Get cache stats
stats = cache.get_stats()
print(f"Cache stats: {stats}")