"""
Genome 🧬 data-structure module for prompt mutation.

This module handles:
- Initializing and mutating genomes(blueprint for prompts).
- Parsing genomes into valid prompts directly feedable to the LLM.
- Decoding, validating and comparing genomes.
"""
from .decoder import GenomeDecoder
from .genome import Genome
from .validator import GenomeValidator

__all__= [
    'Genome',
    'GenomeDecoder',
    'GenomeValidator',
]