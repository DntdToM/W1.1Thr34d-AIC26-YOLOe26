import os
import time
import logging
from typing import List, Dict, Any
import numpy as np
import torch
import scipy.sparse as sp
import sys

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.preprocessing.embedding_gen import EmbeddingGenerator
from src.database.vector_db import HybridVectorDB
from src.utils.lexicon_parser import load_lexicon, extract_canonical_terms

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("FastRetrieval")


class FastRetrievalPipeline:
    def __init__(self, config_path: str = "config.yaml"):
        logger.info("Initializing FastRetrievalPipeline...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load embedding models
        self.emb_gen = EmbeddingGenerator(config_path)
        
        # Load Hybrid DB
        self.db = HybridVectorDB(config_path)
        self.db.load_indices()
        
        # Preload PyTorch sparse tensor to GPU
        self.sparse_db_tensor = self.db.get_sparse_tensor()
        
        # Load Canonical Lexicon
        lexicon_path = os.path.join(os.path.dirname(__file__), "../utils/canonical_lexicon.json")
        self.lexicon_cache = load_lexicon(lexicon_path)

    def query(self, text_query: str, top_k: int = 100, weights: tuple = (0.5, 0.3, 0.2)) -> List[Dict[str, Any]]:
        """
        Execute late fusion retrieval.
        Weights: (vision_weight, text_dense_weight, text_sparse_weight)
        """
        if not self.db.mapping:
            logger.warning("Database mapping is empty.")
            return []

        ntotal = len(self.db.mapping)
        w_v, w_sd, w_sl = weights
        
        # 1. Embed Query
        logger.info(f"Embedding query: '{text_query}'")
        q_v = self.emb_gen.get_image_embeddings_batch([text_query])  # SigLIP can embed text to vision space
        
        # Separate Dense and Sparse inputs
        dense_input = text_query
        canonical_terms = extract_canonical_terms(text_query, self.lexicon_cache)
        sparse_input = f"{text_query} {' '.join(canonical_terms)}"
        
        q_t_dense_dict = self.emb_gen.get_text_embedding(dense_input)
        q_t_sparse_dict = self.emb_gen.get_text_embedding(sparse_input)
        
        q_t_dense = q_t_dense_dict["dense"].reshape(1, -1)
        q_t_sparse = q_t_sparse_dict["sparse"] # scipy.sparse.csr_matrix (1, vocab)

        # We will retrieve top-K candidates from FAISS, but for full SpMM we get all scores
        # Actually, since FAISS is very fast, we can retrieve Top-K=2000 from each
        # But wait, FAISS `search` only returns top-K. To get scores for all, we can query Top-K = ntotal if ntotal is small
        search_k = min(10000, ntotal)
        
        # FAISS Vision
        D_v, I_v = self.db.vision_index.search(q_v, search_k)
        # FAISS Text Dense
        D_t, I_t = self.db.text_dense_index.search(q_t_dense, search_k)
        
        # PyTorch SpMM for Lexical
        # q_t_sparse to torch COO
        coo = q_t_sparse.tocoo()
        q_sparse_tensor = torch.sparse_coo_tensor(
            np.vstack((coo.row, coo.col)), 
            coo.data, 
            size=coo.shape, 
            dtype=torch.float32
        ).to(self.device)
        
        if self.sparse_db_tensor is not None:
            # S_lexical = Q_sparse @ D_sparse.T
            # D_sparse is (N, Vocab), Q_sparse is (1, Vocab)
            # We want (1, N)
            # Actually torch.sparse.mm doesn't support sparse @ sparse transpose easily.
            # Convert query to dense since it's just 1xVocab
            q_dense = q_sparse_tensor.to_dense() # (1, 250002)
            # S_lexical = q_dense @ db_tensor.T
            # Wait, db_tensor is (N, Vocab). db_tensor @ q_dense.T -> (N, 1)
            s_lexical = torch.sparse.mm(self.sparse_db_tensor, q_dense.T).squeeze(1).cpu().numpy()
        else:
            s_lexical = np.zeros(ntotal, dtype=np.float32)

        # Union of indices from Vision and Dense Text
        candidate_indices = np.unique(np.concatenate((I_v[0], I_t[0])))
        candidate_indices = candidate_indices[candidate_indices >= 0]
        
        # Initialize final scores
        # For missing scores in FAISS top-K, they are implicitly 0 (or low).
        # We can reconstruct a dense score array for FAISS if we map them
        s_vision = np.zeros(ntotal, dtype=np.float32)
        s_vision[I_v[0]] = D_v[0]
        
        s_text_dense = np.zeros(ntotal, dtype=np.float32)
        s_text_dense[I_t[0]] = D_t[0]
        
        # Combine scores for candidates
        final_scores = (w_v * s_vision) + (w_sd * s_text_dense) + (w_sl * s_lexical)
        
        # Sort candidates
        sorted_cand_idx = candidate_indices[np.argsort(final_scores[candidate_indices])[::-1]]
        top_candidates = sorted_cand_idx[:top_k * 2] # Get 2x for temporal reranking
        
        # Build raw results
        raw_results = []
        for idx in top_candidates:
            meta = dict(self.db.mapping[idx])
            meta["score"] = float(final_scores[idx])
            meta["s_v"] = float(s_vision[idx])
            meta["s_sd"] = float(s_text_dense[idx])
            meta["s_sl"] = float(s_lexical[idx])
            raw_results.append(meta)
            
        # Temporal Reranking (Two-pointer window boost)
        reranked_results = self._temporal_rerank(raw_results, window_ms=3000, boost_val=0.1)
        
        return reranked_results[:top_k]

    def _temporal_rerank(self, results: List[Dict[str, Any]], window_ms: int = 3000, boost_val: float = 0.1) -> List[Dict[str, Any]]:
        """Boost scores if multiple high-scoring frames appear in the same temporal window for a video."""
        # Group by video_id
        video_groups = {}
        for r in results:
            vid = r["video_id"]
            if vid not in video_groups:
                video_groups[vid] = []
            video_groups[vid].append(r)
            
        for vid, frames in video_groups.items():
            frames.sort(key=lambda x: x["timestamp_ms"])
            
            # Simple O(N^2) for small N per video
            n = len(frames)
            for i in range(n):
                for j in range(i+1, n):
                    if frames[j]["timestamp_ms"] - frames[i]["timestamp_ms"] <= window_ms:
                        # Both frames are close, boost both scores slightly
                        frames[i]["score"] += boost_val
                        frames[j]["score"] += boost_val
                    else:
                        break # sorted, so we can break early
                        
        # Re-sort all by updated score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

if __name__ == "__main__":
    pipeline = FastRetrievalPipeline()
    res = pipeline.query("người đạp xe đạp", top_k=5)
    for r in res:
        print(f"[{r['video_id']} - {r['timestamp_ms']}ms] Score: {r['score']:.4f} | OCR: {r['ocr_text']} | ASR: {r['asr_text']} | Obj: {r['detected_objects']}")
