"""
Embeddings Utility Module

Provides text embedding functionality using sentence-transformers.
"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wrapper for sentence-transformers models.

    Provides caching and batch processing for text embeddings.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize embedding model.

        Args:
            model_name: HuggingFace model name
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.device = device
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device
            )
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text

        Returns:
            Embedding vector (numpy array)
        """
        return self.model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts
            batch_size: Batch size for processing

        Returns:
            Array of embeddings (shape: [len(texts), embedding_dim])
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False
        )

    def cosine_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score (0-1)
        """
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

        return float(similarity)

    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        emb1 = self.embed(text1)
        emb2 = self.embed(text2)
        return self.cosine_similarity(emb1, emb2)

    def similarity_batch(
        self,
        query: str,
        candidates: List[str]
    ) -> List[float]:
        """
        Compute similarity between query and multiple candidates.

        Args:
            query: Query text
            candidates: List of candidate texts

        Returns:
            List of similarity scores
        """
        query_emb = self.embed(query)
        candidate_embs = self.embed_batch(candidates)

        similarities = []
        for candidate_emb in candidate_embs:
            sim = self.cosine_similarity(query_emb, candidate_emb)
            similarities.append(sim)

        return similarities

    def get_embedding_dim(self) -> int:
        """
        Get embedding dimension.

        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()


class HFTextVectorizer:
    """
    Alternative text vectorizer compatible with RedisVL.
    Wrapper around EmbeddingModel for compatibility.
    """

    def __init__(self, model_name: str):
        """
        Initialize vectorizer.

        Args:
            model_name: Model name
        """
        self.embedding_model = EmbeddingModel(model_name)

    def embed(self, text: str) -> List[float]:
        """
        Embed text and return as list.

        Args:
            text: Input text

        Returns:
            Embedding as list of floats
        """
        embedding = self.embedding_model.embed(text)
        return embedding.tolist()

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts.

        Args:
            texts: List of texts

        Returns:
            List of embeddings
        """
        embeddings = self.embedding_model.embed_batch(texts)
        return embeddings.tolist()
