import uuid
from datetime import datetime
from app.services.chat_service import ChatService

class Chat:
    """Model representing a chat conversation"""
    
    def __init__(self, chat_id=None, title=None, user_id=None, 
                 created_at=None, updated_at=None, message_count=0):
        """Initialize a chat object"""
        self.chat_id = chat_id or str(uuid.uuid4())
        self.title = title
        self.user_id = user_id
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or self.created_at
        self.message_count = message_count
        self.service = ChatService()
    
    @classmethod
    def create(cls, title=None, user_id=None):
        """Create a new chat in the database"""
        service = ChatService()
        chat_id = service.create_chat(title=title, user_id=user_id)
        if chat_id:
            return cls.get(chat_id)
        return None
    
    @classmethod
    def get(cls, chat_id):
        """Get a chat by ID"""
        service = ChatService()
        chat_data = service.get_chat(chat_id)
        if chat_data:
            return cls(
                chat_id=chat_data['chat_id'],
                title=chat_data['title'],
                user_id=chat_data.get('user_id'),
                created_at=chat_data.get('created_at'),
                updated_at=chat_data.get('updated_at'),
                message_count=chat_data.get('message_count', 0)
            )
        return None
    
    @classmethod
    def get_recent(cls, limit=10, user_id=None):
        """Get recent chats"""
        service = ChatService()
        chats_data = service.get_recent_chats(limit=limit, user_id=user_id)
        return [cls(
            chat_id=chat_data['chat_id'],
            title=chat_data['title'],
            user_id=chat_data.get('user_id'),
            created_at=chat_data.get('created_at'),
            updated_at=chat_data.get('updated_at'),
            message_count=chat_data.get('message_count', 0)
        ) for chat_data in chats_data]
    
    def add_message(self, role, content, metadata=None):
        """Add a message to the chat"""
        message_id = self.service.add_message(
            self.chat_id, role, content, metadata
        )
        if message_id:
            self.message_count += 1
            self.updated_at = datetime.utcnow()
            return message_id
        return None
    
    def get_messages(self, limit=50, before_id=None):
        """Get messages from the chat"""
        return self.service.get_chat_messages(
            self.chat_id, limit=limit, before_id=before_id
        )
    
    def get_context(self, message_limit=10):
        """Get chat context for a follow-up query"""
        return self.service.get_chat_context(
            self.chat_id, message_limit=message_limit
        )
    
    def update_title(self, new_title):
        """Update the chat title"""
        success = self.service.update_chat_title(self.chat_id, new_title)
        if success:
            self.title = new_title
        return success
    
    def generate_title(self, llm_service):
        """Generate a title for the chat using LLM"""
        title = self.service.generate_chat_title(self.chat_id, llm_service)
        if title:
            self.title = title
        return title
    
    def delete(self):
        """Delete the chat and all its messages"""
        return self.service.delete_chat(self.chat_id)
    
    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            'chat_id': self.chat_id,
            'title': self.title,
            'user_id': self.user_id,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'message_count': self.message_count
        }