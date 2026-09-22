import os
import json
from datetime import datetime
from uuid import uuid4

class BaseModel:
    """Base class for file-based persistence models"""
    STORAGE_DIR = None
    
    @classmethod
    def create_storage(cls):
        """Create storage directory if it doesn't exist"""
        if cls.STORAGE_DIR:
            os.makedirs(cls.STORAGE_DIR, exist_ok=True)
    
    def save(self):
        """Save model to file storage"""
        self.create_storage()
        file_path = os.path.join(self.STORAGE_DIR, f"{self.id}.json")
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f)
        return self
    
    @classmethod
    def get_all(cls, filter_field=None, filter_value=None):
        """Get all instances, optionally filtered by a field"""
        cls.create_storage()
        items = []
        for filename in os.listdir(cls.STORAGE_DIR):
            if filename.endswith('.json'):
                with open(os.path.join(cls.STORAGE_DIR, filename), 'r') as f:
                    data = json.load(f)
                    if filter_field is None or data.get(filter_field) == filter_value:
                        items.append(cls(**data))
        return items
    
    @classmethod
    def get(cls, id):
        """Get instance by id"""
        cls.create_storage()
        file_path = os.path.join(cls.STORAGE_DIR, f"{id}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                return cls(**data)
        return None
    
    def delete(self):
        """Delete instance"""
        file_path = os.path.join(self.STORAGE_DIR, f"{self.id}.json")
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False