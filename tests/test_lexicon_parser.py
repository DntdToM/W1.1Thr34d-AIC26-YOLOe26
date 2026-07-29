import pytest
import os
import json
import tempfile
from src.utils.lexicon_parser import normalize_text, load_lexicon, extract_canonical_terms

@pytest.fixture
def sample_lexicon_path():
    data = {
        "metadata": {"version": "1.0.0"},
        "entities": {
            "car": {
                "enabled": True,
                "aliases": {"vi": ["ô tô", "xe hơi"]},
                "controlled_variants": {"vi": ["sedan"]}
            },
            "drone": {
                "enabled": True,
                "aliases": {"vi": ["máy bay không người lái", "flycam"]},
            },
            "airplane": {
                "enabled": True,
                "aliases": {"vi": ["máy bay", "phi cơ"]}
            },
            "disabled_obj": {
                "enabled": False,
                "aliases": {"vi": ["vật bị tắt"]}
            }
        }
    }
    
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.json', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        temp_path = f.name
        
    yield temp_path
    
    os.remove(temp_path)

def test_normalize_text():
    # Unicode NFC test
    assert normalize_text("x\u0065\u0301") == "xé"
    # Lowercase & strip
    assert normalize_text("  XE HƠI  ") == "xe hơi"
    # Punctuation to space and collapse
    assert normalize_text("tìm xe hơi, và máy bay!!!") == "tìm xe hơi và máy bay"

def test_feature_flag(sample_lexicon_path):
    lexicon = load_lexicon(sample_lexicon_path)
    terms = extract_canonical_terms("tìm vật bị tắt", lexicon)
    assert "disabled_obj" not in terms

def test_exact_alias(sample_lexicon_path):
    lexicon = load_lexicon(sample_lexicon_path)
    terms = extract_canonical_terms("xe hơi", lexicon)
    assert "car" in terms

def test_controlled_variant(sample_lexicon_path):
    lexicon = load_lexicon(sample_lexicon_path)
    terms = extract_canonical_terms("tôi lái sedan", lexicon)
    assert "car" in terms

def test_longest_match(sample_lexicon_path):
    lexicon = load_lexicon(sample_lexicon_path)
    terms = extract_canonical_terms("máy bay không người lái đang bay", lexicon)
    assert "drone" in terms
    assert "airplane" not in terms # Should not match "máy bay" due to overlap

def test_boundary_safety(sample_lexicon_path):
    lexicon = load_lexicon(sample_lexicon_path)
    # "car" should not match in "carpet"
    terms = extract_canonical_terms("carpet", lexicon)
    assert "car" not in terms
    # Verify exact match still works
    terms = extract_canonical_terms("my car", lexicon)
    assert "car" in terms

def test_stability_regression(sample_lexicon_path):
    lexicon = load_lexicon(sample_lexicon_path)
    # Check stable assignment over multiple calls
    res1 = extract_canonical_terms("xe hơi", lexicon)
    res2 = extract_canonical_terms("ô tô", lexicon)
    res3 = extract_canonical_terms("sedan", lexicon)
    assert res1 == ["car"]
    assert res2 == ["car"]
    assert res3 == ["car"]

def test_vietnamese_word_boundary(sample_lexicon_path):
    lexicon = load_lexicon(sample_lexicon_path)
    # Test if punctuation padding works correctly for Vietnamese
    terms = extract_canonical_terms("tìm xe hơi, đồ cũ", lexicon)
    assert "car" in terms
