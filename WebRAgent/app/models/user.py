import os
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from bson.objectid import ObjectId
import hashlib

# MongoDB connection
client = MongoClient(os.environ.get('MONGODB_URI', 'mongodb://localhost:27017'))
db = client.ragapp
users_collection = db.users

class User(UserMixin):
    def __init__(self, id, username, password_hash, is_admin=False, **kwargs):
        self.id = str(id)
        self.username = username
        self.password_hash = password_hash
        self.is_admin = is_admin
        
        # Add any additional fields
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def get(cls, id):
        """Get user by ID"""
        if not id:
            return None
            
        # Handle transition from string IDs to ObjectIds
        try:
            # If ID looks like a legacy string ID (from in-memory store)
            if id == '1':
                # Try to get admin user by username first
                admin_username = os.environ.get('ADMIN_USERNAME', 'admin')
                user_data = users_collection.find_one({'username': admin_username})
                if user_data:
                    user_data['id'] = user_data.pop('_id')
                    return cls(**user_data)
                return None
                
            # Otherwise try to convert to ObjectId
            user_data = users_collection.find_one({'_id': ObjectId(id)})
            if user_data:
                user_data['id'] = user_data.pop('_id')
                return cls(**user_data)
        except:
            # If conversion fails, return None
            return None
            
        return None
    
    @classmethod
    def get_by_username(cls, username):
        """Get user by username"""
        user_data = users_collection.find_one({'username': username})
        if user_data:
            user_data['id'] = user_data.pop('_id')
            return cls(**user_data)
        return None
        
    @classmethod
    def create_user(cls, username, password, is_admin=False):
        """Create a new user"""
        # Check if username already exists
        if cls.get_by_username(username):
            return None, "Username already exists"
            
        # Create new user document
        user_doc = {
            'username': username,
            'password_hash': generate_password_hash(password),
            'is_admin': is_admin
        }
        
        result = users_collection.insert_one(user_doc)
        user_doc['id'] = result.inserted_id
        return cls(**user_doc), "User created successfully"
    
    @classmethod
    def get_all(cls):
        """Get all users"""
        users = []
        for user_data in users_collection.find():
            user_data['id'] = user_data.pop('_id')
            users.append(cls(**user_data))
        return users
        
    @classmethod
    def delete(cls, user_id):
        """Delete a user"""
        result = users_collection.delete_one({'_id': ObjectId(user_id)})
        return result.deleted_count > 0
        
    @classmethod
    def initialize_admin(cls):
        """Initialize admin user from environment variables"""
        admin_username = os.environ.get('ADMIN_USERNAME', 'admin')
        admin_password = os.environ.get('ADMIN_PASSWORD')
        
        # Skip if no admin password set
        if not admin_password:
            print("Warning: No ADMIN_PASSWORD set in environment. Using default password.")
            admin_password = 'admin123'  # Default for development only
            
        # Check if admin user already exists
        admin = cls.get_by_username(admin_username)
        if admin:
            # Update admin password if it's changed (using hash comparison)
            if not check_password_hash(admin.password_hash, admin_password):
                users_collection.update_one(
                    {'username': admin_username},
                    {'$set': {'password_hash': generate_password_hash(admin_password)}}
                )
            return
            
        # Create admin user if it doesn't exist
        cls.create_user(
            username=admin_username,
            password=admin_password,
            is_admin=True
        )
        print(f"Admin user '{admin_username}' created successfully.")
    
    def check_password(self, password):
        """Check password using secure verification"""
        return check_password_hash(self.password_hash, password)
        
    def to_dict(self):
        """Convert user to dictionary for API responses"""
        return {
            'id': self.id,
            'username': self.username,
            'is_admin': self.is_admin
        }