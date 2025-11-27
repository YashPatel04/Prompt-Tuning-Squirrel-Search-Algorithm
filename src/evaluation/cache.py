import json
import hashlib
from typing import Optional, Dict
from pathlib import Path

class PromptCache:
    """
    Cache LLM responses to avoid redundant API calls.
    """
    def __init__(self, cache_dir='data/cache', use_disk=True):
        """
        Initialize cache.
        Args:
            cache_dir: Directory to store cache files
            use_disk: Whether to persist cache to disk
        """
        self.memory_cache: Dict[str, str] = {}
        self.cache_dir = Path(cache_dir)
        self.use_disk = use_disk
        
        if use_disk:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_disk_cache()
    
    def _hash_prompt(self, prompt):
        """Generate hash of prompt for cache key"""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def get(self, prompt):
        """
        Retrieve cached response for prompt.
        Args:
            prompt: Input prompt
        Returns:
            Cached response or None
        """
        cache_key = self._hash_prompt(prompt)
        # Check memory cache first
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        return None
    
    def set(self, prompt, response):
        """
        Cache a prompt-response pair.
        Args:
            prompt: Input prompt
            response: LLM response
        """
        cache_key = self._hash_prompt(prompt)
        # Store in memory
        self.memory_cache[cache_key] = response
        # Store on disk
        if self.use_disk:
            self._save_to_disk(cache_key, prompt, response)
    
    def _save_to_disk(self, cache_key, prompt, response):
        """Save cache entry to disk"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        data = {
            'prompt': prompt,
            'response': response
        }
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cache to disk: {e}")
    
    def _load_disk_cache(self):
        """Load cached responses from disk"""
        if not self.cache_dir.exists():
            return
        for cache_file in self.cache_dir.glob('*.json'):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    cache_key = cache_file.stem
                    self.memory_cache[cache_key] = data['response']
            except Exception as e:
                print(f"Warning: Could not load cache file {cache_file}: {e}")
    
    def clear(self):
        """Clear all cached entries"""
        self.memory_cache.clear()
        print("Cache cleared")
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'cached_entries': len(self.memory_cache),
            'cache_size_mb': sum(
                len(v.encode('utf-8')) for v in self.memory_cache.values()
            ) / (1024 * 1024)
        }