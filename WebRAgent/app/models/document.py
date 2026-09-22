import os
from datetime import datetime
from uuid import uuid4
from app.models.base import BaseModel

class Document(BaseModel):
    """Model for documents within collections"""
    STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'documents')
    
    def __init__(self, id=None, collection_id=None, title=None, content=None, 
                 file_path=None, metadata=None, created_at=None):
        self.id = id or str(uuid4())
        self.collection_id = collection_id
        self.title = title
        self.content = content
        self.file_path = file_path
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.now().isoformat()
    
    def to_dict(self):
        """Convert document to dictionary"""
        return {
            'id': self.id,
            'collection_id': self.collection_id,
            'title': self.title,
            'content': self.content,
            'file_path': self.file_path,
            'metadata': self.metadata,
            'created_at': self.created_at
        }
    
    @classmethod
    def get_all(cls, collection_id=None):
        """Get all documents, optionally filtered by collection_id"""
        return super().get_all(filter_field='collection_id', filter_value=collection_id)