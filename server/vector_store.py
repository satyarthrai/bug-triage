# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Vector Store Implementation for Bug Triage.

Provides semantic search capabilities using FAISS and Sentence Transformers
to identify historical duplicates based on bug descriptions.
"""

from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class BugVectorStore:
    """
    Local Vector Database for semantic bug similarity matching.

    This class handles the embedding of historical bug descriptions and 
    provides a Top-K search interface for the RL agent during Phase 2.
    """

    def __init__(self, dataset: List[Dict[str, Any]]):
        """
        Initialize the vector store and index the historical dataset.

        Args:
            dataset: The historical bug dataset containing 'user_description' keys.
        """
        # Load the embedding model (Tiny and efficient for CPU/Docker environments)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.bugs = dataset
        
        # Extract descriptions for embedding
        self.descriptions = [b.get("user_description", "") for b in self.bugs]
        
        # Generate embeddings and convert to float32 for FAISS compatibility
        self.embeddings = self.model.encode(self.descriptions).astype('float32')
        
        # Initialize a flat L2 index (Exact search)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        
    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Perform a semantic search for the given query.

        Args:
            query: The natural language search string or bug description.
            k: Number of top matches to return.

        Returns:
            A list of dictionary results including bug metadata and similarity scores.
        """
        if not query:
            return []

        # Generate vector for the search query
        query_vector = self.model.encode([query]).astype('float32')
        
        # Search the index for the Top-K nearest neighbors
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            # FAISS might return -1 if not enough results are found
            if idx == -1:
                continue
                
            bug = self.bugs[idx]
            
            # Convert L2 distance to a 0-1 similarity score for the RL Agent
            # Higher score = more similar
            similarity = round(float(1 / (1 + distances[0][i])), 3)
            
            results.append({
                "bug_id": str(bug.get("id", "unknown")),
                "description": bug.get("user_description", ""),
                "repo": bug.get("route_to", "unknown"),
                "similarity": similarity
            })
                
        return results