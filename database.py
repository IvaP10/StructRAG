from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SparseVector,
    PayloadSchemaType
)
from typing import List, Dict, Any, Optional
from uuid import UUID
import numpy as np
import config
from models import Chunk
import logging

logger = logging.getLogger(__name__)


class EnhancedVectorDatabase:
    
    def __init__(self):
        if config.QDRANT_PATH:
            # Embedded mode: no server process, storage is a local directory.
            # Checked first so it wins over any stale host/port left in the env.
            logger.info(f"Qdrant embedded at {config.QDRANT_PATH}")
            self.client = QdrantClient(path=config.QDRANT_PATH)
        elif config.QDRANT_API_KEY:
            self.client = QdrantClient(
                url=f"https://{config.QDRANT_HOST}",
                api_key=config.QDRANT_API_KEY
            )
        else:
            self.client = QdrantClient(
                host=config.QDRANT_HOST,
                port=config.QDRANT_PORT,
                timeout=60
            )


        self.collection_name = config.QDRANT_COLLECTION
        self.embedding_dimension = config.EMBEDDING_DIMENSION
        
        self._ensure_collection()
    
    def _ensure_collection(self):
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                try:
                    self._create_collection()
                except Exception as create_err:
                    if "File exists" in str(create_err) or "already exists" in str(create_err).lower():
                        logger.warning(
                            f"Stale storage found for '{self.collection_name}'. "
                            f"Cleaning up and recreating..."
                        )
                        try:
                            self.client.delete_collection(self.collection_name)
                            import time
                            time.sleep(1.0)
                        except Exception:  # nosec B110 - best effort; the create below reports real failures
                            pass
                        self._create_collection()
                    else:
                        raise
                
        except Exception:
            raise

    def reset_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
            import time
            time.sleep(1.0)
        except Exception:  # nosec B110 - nothing to delete on a fresh collection
            pass
        self._create_collection()
    
    def _create_collection(self):
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE,
                    on_disk=False
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(
                        on_disk=False
                    )
                )
            }
        )
        self._create_payload_indexes()

    def _create_payload_indexes(self):
        """Index the fields we filter on.

        Without these, Qdrant falls back to a full scan for every filtered
        search, which gets slow as sessions accumulate in the shared collection.
        """
        for field in ("session_id", "document_id"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD
                )
            except Exception as e:
                # Non-fatal: searches still work, just unindexed.
                logger.warning(f"Could not create payload index on '{field}': {e}")
    
    def index_chunks(
        self,
        chunks: List[Chunk],
        dense_embeddings: np.ndarray,
        sparse_vectors: List[Dict[int, float]]
    ):
        if len(chunks) != len(dense_embeddings) or len(chunks) != len(sparse_vectors):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks, {len(dense_embeddings)} embeddings, "
                f"{len(sparse_vectors)} sparse vectors"
            )
        
        child_indices = [i for i, c in enumerate(chunks) if c.is_child]
        child_chunks = [chunks[i] for i in child_indices]
        child_dense = [dense_embeddings[i] for i in child_indices]
        child_sparse = [sparse_vectors[i] for i in child_indices]
        
        if not child_chunks:
            return
        
        points = []
        for chunk, dense_emb, sparse_vec in zip(child_chunks, child_dense, child_sparse):
            point = self._create_point(chunk, dense_emb, sparse_vec)
            points.append(point)
        
        try:
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            
        except Exception as e:
            logger.error(f"Indexing error: {str(e)}")
            raise
    
    def _create_point(
        self,
        chunk: Chunk,
        dense_embedding: np.ndarray,
        sparse_vector: Dict[int, float]
    ) -> PointStruct:
        if isinstance(dense_embedding, np.ndarray):
            dense_list = dense_embedding.tolist()
        else:
            dense_list = list(dense_embedding)
        
        sparse_indices = list(sparse_vector.keys())
        sparse_values = list(sparse_vector.values())
        
        point = PointStruct(
            id=str(chunk.id),
            vector={
                "dense": dense_list,
                "sparse": SparseVector(
                    indices=sparse_indices,
                    values=sparse_values
                )
            },
            payload={
                "chunk_id": str(chunk.id),
                "document_id": str(chunk.document_id),
                # Stamped into chunk.metadata by the caller before indexing, the
                # same way source_filename is. Absent in single-user CLI mode,
                # where there is nothing to isolate.
                "session_id": chunk.metadata.get("session_id"),
                "parent_id": str(chunk.parent_id) if chunk.parent_id else None,
                "text": chunk.text,
                "chunk_type": chunk.chunk_type.value,
                "format_type": chunk.format_type.value,
                "page_number": chunk.page_number,
                "bbox": chunk.bbox.dict() if chunk.bbox else None,
                "token_count": chunk.token_count,
                "source_filename": chunk.metadata.get("source_filename", "unknown"),
                "metadata": chunk.metadata
            }
        )
        
        return point
    
    def search(
        self,
        dense_embedding: np.ndarray,
        sparse_vector: Dict[int, float],
        document_id: Optional[UUID] = None,
        top_k: int = 50,
        use_dense: bool = True,
        use_sparse: bool = True,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        logger.info(f"Searching for top {top_k} chunks")
        logger.info(f"Search mode: dense={use_dense}, sparse={use_sparse}")
        
        try:
            query_filter = self._build_filter(document_id, metadata_filter)
            
            if use_dense and use_sparse:
                results = self._hybrid_search(
                    dense_embedding, sparse_vector, query_filter, top_k
                )
            elif use_dense:
                results = self._dense_search(
                    dense_embedding, query_filter, top_k
                )
            elif use_sparse:
                results = self._sparse_search(
                    sparse_vector, query_filter, top_k
                )
            else:
                logger.warning("Both dense and sparse disabled, returning empty results")
                return []
            
            formatted = self._format_results(results)
            logger.info(f"Found {len(formatted)} results")
            
            return formatted
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise
    
    def _build_filter(
        self,
        document_id: Optional[UUID] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None
    ) -> Optional[Filter]:
        conditions = []

        if document_id is not None:
            conditions.append(
                FieldCondition(
                    key="document_id",
                    match=MatchValue(value=str(document_id))
                )
            )

        # Multi-tenant isolation. In server mode every search passes a
        # session_id, so one visitor can never retrieve another's chunks even
        # though all sessions share a single collection.
        if session_id is not None:
            conditions.append(
                FieldCondition(
                    key="session_id",
                    match=MatchValue(value=str(session_id))
                )
            )

        if metadata_filter:
            for key, value in metadata_filter.items():
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value)
                    )
                )
        
        return Filter(must=conditions) if conditions else None
    
    def _dense_search(
        self,
        dense_embedding: np.ndarray,
        query_filter: Filter,
        top_k: int
    ) -> List:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_embedding.tolist(),
            using="dense",
            query_filter=query_filter,
            limit=top_k,
            with_payload=True
        ).points
        
        return results
    
    def _sparse_search(
        self,
        sparse_vector: Dict[int, float],
        query_filter: Filter,
        top_k: int
    ) -> List:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=SparseVector(
                indices=list(sparse_vector.keys()),
                values=list(sparse_vector.values())
            ),
            using="sparse",
            query_filter=query_filter,
            limit=top_k,
            with_payload=True
        ).points
        
        return results
    
    def _hybrid_search(
        self,
        dense_embedding: np.ndarray,
        sparse_vector: Dict[int, float],
        query_filter: Filter,
        top_k: int
    ) -> List:
        logger.debug("Performing hybrid search with RRF")
        
        fetch_k = min(top_k * 3, 150)
        
        dense_results = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_embedding.tolist(),
            using="dense",
            query_filter=query_filter,
            limit=fetch_k,
            with_payload=True
        ).points
        
        sparse_results = self.client.query_points(
            collection_name=self.collection_name,
            query=SparseVector(
                indices=list(sparse_vector.keys()),
                values=list(sparse_vector.values())
            ),
            using="sparse",
            query_filter=query_filter,
            limit=fetch_k,
            with_payload=True
        ).points
        
        fused_results = self._reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            top_k,
            k=config.RRF_K_PARAM
        )
        
        return fused_results
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List,
        sparse_results: List,
        top_k: int,
        k: int = 60
    ) -> List:
        scores = {}
        
        dense_weight = config.DENSE_WEIGHT
        sparse_weight = config.SPARSE_WEIGHT
        
        for rank, point in enumerate(dense_results, start=1):
            point_id = point.id
            rrf_score = 1.0 / (k + rank)
            scores[point_id] = scores.get(point_id, 0) + (dense_weight * rrf_score)
        
        for rank, point in enumerate(sparse_results, start=1):
            point_id = point.id
            rrf_score = 1.0 / (k + rank)
            scores[point_id] = scores.get(point_id, 0) + (sparse_weight * rrf_score)
        
        all_points = {p.id: p for p in dense_results}
        all_points.update({p.id: p for p in sparse_results})
        
        ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
        
        ranked_points = []
        for point_id in ranked_ids:
            point = all_points[point_id]
            point.score = scores[point_id]
            ranked_points.append(point)
        
        logger.debug(f"RRF fused {len(dense_results)} dense + {len(sparse_results)} sparse "
                    f"-> {len(ranked_points)} results")
        
        return ranked_points
    
    def _format_results(self, results: List) -> List[Dict[str, Any]]:
        formatted = []
        
        for result in results:
            formatted.append({
                "chunk_id": result.payload["chunk_id"],
                "parent_id": result.payload.get("parent_id"),
                "text": result.payload["text"],
                "chunk_type": result.payload["chunk_type"],
                "format_type": result.payload.get("format_type", "plain"),
                "page_number": result.payload["page_number"],
                "source_filename": result.payload.get("source_filename", "unknown"),
                "bbox": result.payload.get("bbox"),
                "token_count": result.payload.get("token_count", 0),
                "score": float(result.score) if hasattr(result, 'score') else 0.0,
                "metadata": result.payload.get("metadata", {})
            })
        
        return formatted
    
    def delete_document(self, document_id: UUID):
        logger.info(f"Deleting document {document_id}")

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=str(document_id))
                        )
                    ]
                )
            )
            logger.info(f"Document {document_id} deleted successfully")

        except Exception as e:
            logger.error(f"Deletion error: {str(e)}")
            raise

    def delete_session(self, session_id: str):
        """Drop every chunk belonging to one session.

        This is how the server reclaims space when a session expires. It
        replaces the CLI's reset_collection(), which wipes the whole collection
        and would destroy other visitors' documents.
        """
        logger.info(f"Deleting session {session_id}")

        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="session_id",
                            match=MatchValue(value=str(session_id))
                        )
                    ]
                )
            )
        except Exception as e:
            logger.error(f"Session deletion error: {str(e)}")
            raise
    
    def hybrid_search(
        self,
        document_id: Optional[UUID] = None,
        dense_vector: np.ndarray = None,
        sparse_vector: Dict[int, float] = None,
        limit: int = 50,
        dense_weight: float = 0.65,
        sparse_weight: float = 0.35,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        query_filter = self._build_filter(document_id, session_id=session_id)
        results = self._hybrid_search(dense_vector, sparse_vector, query_filter, limit)
        return self._format_results(results)

    def dense_search(
        self,
        document_id: Optional[UUID] = None,
        dense_vector: np.ndarray = None,
        limit: int = 50,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        query_filter = self._build_filter(document_id, session_id=session_id)
        results = self._dense_search(dense_vector, query_filter, limit)
        return self._format_results(results)

    def sparse_search(
        self,
        document_id: Optional[UUID] = None,
        sparse_vector: Dict[int, float] = None,
        limit: int = 50,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        query_filter = self._build_filter(document_id, session_id=session_id)
        results = self._sparse_search(sparse_vector, query_filter, limit)
        return self._format_results(results)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                "total_points": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "status": collection_info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}


# Create the singleton instance with proper error handling
class _LazyVectorDB:
    """Defers Qdrant connection until first attribute access."""
    def __init__(self):
        self._instance = None
        self._init_error = None

    def _ensure(self):
        if self._instance is None and self._init_error is None:
            try:
                self._instance = EnhancedVectorDatabase()
            except Exception as e:
                self._init_error = e
                logger.error(f"Failed to initialize vector database: {str(e)}")
                logger.error("Please set up Qdrant before running the application.")

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        self._ensure()
        if self._instance is None:
            raise RuntimeError(f"Vector database not available: {self._init_error}")
        return getattr(self._instance, name)

vector_db = _LazyVectorDB()