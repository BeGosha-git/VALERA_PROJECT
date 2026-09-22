import os
from datetime import datetime
from uuid import uuid4
from app.models.base import BaseModel

class Collection(BaseModel):
    """Model for document collections"""
    STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'collections')
    
    def __init__(self, id=None, name=None, description=None, created_at=None, 
                 embedding_provider=None, embedding_model=None, embedding_dimensions=None):
        self.id = id or str(uuid4())
        self.name = name
        self.description = description
        self.created_at = created_at or datetime.now().isoformat()
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
    
    def to_dict(self):
        """Convert collection to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at,
            'embedding_provider': self.embedding_provider,
            'embedding_model': self.embedding_model,
            'embedding_dimensions': self.embedding_dimensions
        }