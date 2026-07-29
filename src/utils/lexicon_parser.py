import json
import unicodedata
import re
import os
from typing import List, Dict, Tuple


def normalize_text(text: str) -> str:
    """
    NFC normalization, lowercase, strip, and collapse whitespace.
    Removes common punctuation to allow safe word-boundary matching.
    """
    if not text:
        return ""
    
    # 1. Unicode NFC
    text = unicodedata.normalize('NFC', text)
    
    # 2. Lowercase
    text = text.lower()
    
    # 3. Replace punctuation with space
    # Keep alphanumeric and spaces. Remove things like , . ? ! ; : ( )
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 4. Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def load_lexicon(file_path: str) -> List[Tuple[str, str]]:
    """
    Load canonical_lexicon.json.
    Returns a list of (normalized_alias, canonical_id) sorted by alias length descending.
    This guarantees Longest Match First extraction.
    """
    if not os.path.exists(file_path):
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entities = data.get("entities", {})
    alias_map = {}

    for canonical_id, node in entities.items():
        if not node.get("enabled", True):
            continue
            
        # Add the canonical ID itself
        norm_id = normalize_text(canonical_id)
        if norm_id:
            alias_map[norm_id] = canonical_id
            
        aliases = node.get("aliases", {})
        for lang, terms in aliases.items():
            for term in terms:
                norm_term = normalize_text(term)
                if norm_term:
                    alias_map[norm_term] = canonical_id

        variants = node.get("controlled_variants", {})
        for lang, terms in variants.items():
            for term in terms:
                norm_term = normalize_text(term)
                if norm_term:
                    alias_map[norm_term] = canonical_id

    # Sort by length descending (longest match first)
    # Then by alias string to ensure stable order
    sorted_aliases = sorted(alias_map.items(), key=lambda x: (-len(x[0]), x[0]))
    return sorted_aliases


def extract_canonical_terms(query: str, lexicon: List[Tuple[str, str]]) -> List[str]:
    """
    Extract canonical IDs from query using O(n*m) Padding Trick.
    Guarantees word-boundary safety for Vietnamese Unicode without third-party regex.
    """
    if not query or not lexicon:
        return []

    norm_query = normalize_text(query)
    # Pad query with spaces for exact word boundary match
    padded_query = f" {norm_query} "
    
    query_len = len(padded_query)
    # overlap_mask[i] = True means character at index i is already part of a matched term
    overlap_mask = [False] * query_len
    
    extracted_ids = set()

    for alias, canonical_id in lexicon:
        padded_alias = f" {alias} "
        alias_len = len(padded_alias)
        
        start_idx = 0
        while True:
            # Find occurrence
            idx = padded_query.find(padded_alias, start_idx)
            if idx == -1:
                break
                
            # Check overlap mask
            # We must only match if ALL characters inside the span (except the boundary spaces) 
            # are NOT already consumed.
            # Actual word is from idx + 1 to idx + alias_len - 1
            word_start = idx + 1
            word_end = idx + alias_len - 1
            
            is_overlapped = any(overlap_mask[word_start:word_end])
            
            if not is_overlapped:
                # Valid match
                extracted_ids.add(canonical_id)
                # Mark mask (mark the word part, not the padding space so adjacent words can match)
                for i in range(word_start, word_end):
                    overlap_mask[i] = True
                    
            start_idx = idx + 1

    return list(extracted_ids)


if __name__ == "__main__":
    lex = load_lexicon("canonical_lexicon.json")
    print(f"Loaded {len(lex)} aliases.")
    q = "tìm một chiếc xe hơi, và máy bay không người lái"
    terms = extract_canonical_terms(q, lex)
    print(f"Query: {q}")
    print(f"Extracted: {terms}")
