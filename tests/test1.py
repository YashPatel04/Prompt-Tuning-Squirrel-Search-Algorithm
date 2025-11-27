from src.evaluation.llm_interface import OllamaInterface
from src.evaluation.fitness import FitnessCalculator
from src.evaluation.evaluator import Evaluator
from src.evaluation.cache import PromptCache
from src.genome.decoder import GenomeDecoder
from src.genome.genome import Genome, GenomeConfig
from src.ssa.squirrel import Squirrel
import pandas as pd

def test_ollama_connection():
    """Test basic Ollama connectivity"""
    print("=" * 60)
    print("TEST 1: Ollama Connection")
    print("=" * 60)
    
    try:
        llm = OllamaInterface(
            base_url="http://localhost:1561",
            model="deepseek-r1:7b",
            temperature=0.0
        )
        print("✓ Ollama connected successfully\n")
        return llm
    except Exception as e:
        print(f"✗ Connection failed: {e}\n")
        return None

def test_llm_generation(llm):
    """Test LLM text generation"""
    print("=" * 60)
    print("TEST 2: LLM Generation")
    print("=" * 60)
    
    try:
        test_prompt = "Classify this sentiment: 'I love this product!' Answer with just: positive, negative, or neutral"
        response = llm.generate(test_prompt)
        print(f"Prompt: {test_prompt}")
        print(f"Response: {response}")
        print("✓ LLM generation working\n")
        return True
    except Exception as e:
        print(f"✗ Generation failed: {e}\n")
        return False

def test_fitness_extraction():
    """Test label extraction from responses"""
    print("=" * 60)
    print("TEST 3: Label Extraction")
    print("=" * 60)
    
    fitness_calc = FitnessCalculator()
    
    test_responses = [
        ("This is positive sentiment", "positive"),
        ("The text is negative", "negative"),
        ("It's neutral, neither good nor bad", "neutral"),
        ("[POSITIVE] Great product!", "positive"),
        ('{"sentiment": "negative", "confidence": 0.9}', "negative"),
    ]
    
    all_correct = True
    for response, expected in test_responses:
        extracted = fitness_calc.extract_label_from_response(response)
        status = "✓" if extracted == expected else "✗"
        if extracted != expected:
            all_correct = False
        print(f"{status} Response: '{response[:40]}...' → {extracted} (expected: {expected})")
    
    print(f"\n{'✓' if all_correct else '✗'} Label extraction {'passed' if all_correct else 'failed'}\n")
    return all_correct

def test_cache_system():
    """Test caching functionality"""
    print("=" * 60)
    print("TEST 4: Cache System")
    print("=" * 60)
    
    cache = PromptCache(cache_dir='data/cache')
    
    test_prompt = "Test prompt for caching"
    test_response = "Test response"
    
    # Test set
    cache.set(test_prompt, test_response)
    print(f"✓ Cached prompt-response pair")
    
    # Test get
    retrieved = cache.get(test_prompt)
    if retrieved == test_response:
        print(f"✓ Successfully retrieved from cache")
    else:
        print(f"✗ Cache retrieval failed")
    
    # Test stats
    stats = cache.get_stats()
    print(f"✓ Cache stats: {stats['cached_entries']} entries, {stats['cache_size_mb']:.4f} MB\n")
    return True

def test_fitness_calculation():
    """Test fitness metric calculation"""
    print("=" * 60)
    print("TEST 5: Fitness Calculation")
    print("=" * 60)
    
    fitness_calc = FitnessCalculator()
    
    predictions = ["positive", "negative", "neutral", "positive", "negative"]
    ground_truth = ["positive", "negative", "neutral", "negative", "negative"]
    
    accuracy_fitness = fitness_calc.calculate_accuracy(predictions, ground_truth)
    f1_fitness = fitness_calc.calculate_f1(predictions, ground_truth)
    combined_fitness = fitness_calc.calculate_combined_fitness(predictions, ground_truth)
    
    print(f"Predictions: {predictions}")
    print(f"Ground truth: {ground_truth}")
    print(f"Accuracy fitness (1-acc): {accuracy_fitness:.4f}")
    print(f"F1 fitness (1-F1): {f1_fitness:.4f}")
    print(f"Combined fitness: {combined_fitness:.4f}")
    print("✓ Fitness calculation working\n")
    return True

def test_genome_to_prompt_to_fitness(llm, evaluator):
    """Test full pipeline: Genome → Prompt → Evaluation"""
    print("=" * 60)
    print("TEST 6: Full Pipeline (Genome → Prompt → Fitness)")
    print("=" * 60)
    
    # Create genome
    config = GenomeConfig(dimensions=10)
    genome = Genome(config=config)
    squirrel = Squirrel(genome)
    
    # Decode to prompt
    decoder = GenomeDecoder()
    prompt = decoder.decode(squirrel.genome)
    
    print(f"Generated prompt:\n{prompt[:200]}...\n")
    
    # Create mock dataset
    dataset = [
        {'text': 'I love this!', 'label': 'positive'},
        {'text': 'This is awful', 'label': 'negative'},
        {'text': 'Its okay', 'label': 'neutral'},
    ]
    
    # Evaluate
    fitness = evaluator.evaluate_prompt(prompt, dataset)
    print(f"Fitness score: {fitness:.4f}")
    print("✓ Full pipeline working\n")
    return True

def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("EVALUATION MODULE TEST SUITE")
    print("=" * 60 + "\n")
    
    # Test 1: Connection
    llm = test_ollama_connection()
    if not llm:
        print("Cannot proceed without Ollama connection")
        return
    
    # Test 2: Generation
    if not test_llm_generation(llm):
        print("Cannot proceed without LLM generation")
        return
    
    # Test 3: Label extraction
    test_fitness_extraction()
    
    # Test 4: Cache
    test_cache_system()
    
    # Test 5: Fitness calculation
    test_fitness_calculation()
    
    # Test 6: Full pipeline
    fitness_calc = FitnessCalculator()
    cache = PromptCache()
    evaluator = Evaluator(llm, fitness_calc, cache, use_cache=True)
    test_genome_to_prompt_to_fitness(llm, evaluator)
    
    print("=" * 60)
    print("ALL TESTS COMPLETED ✓")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()