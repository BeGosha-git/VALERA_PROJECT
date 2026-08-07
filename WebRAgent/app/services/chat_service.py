import os
import logging
import uuid
from datetime import datetime
from pymongo import MongoClient
from bson.objectid import ObjectId
from app.services.prompt_template_service import PromptTemplateService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatService:
    """
    Service for managing chat history and conversations using MongoDB
    """
    
    def __init__(self):
        """Initialize Chat service with MongoDB connection"""
        self.mongo_uri = os.environ.get('MONGODB_URI', 'mongodb://mongodb:27017/')
        self.db_name = os.environ.get('MONGODB_DB', 'ragapp')
        self.chats_collection_name = os.environ.get('MONGODB_COLLECTION_CHATS', 'chat_history')
        self.messages_collection_name = os.environ.get('MONGODB_COLLECTION_MESSAGES', 'chat_messages')
        
        # Connect to MongoDB
        try:
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client[self.db_name]
            self.chats = self.db[self.chats_collection_name]
            self.messages = self.db[self.messages_collection_name]
            logger.info(f"Connected to MongoDB at {self.mongo_uri}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            # Using in-memory fallback
            self.client = None
            self.in_memory_chats = []
            self.in_memory_messages = []
            logger.warning("Using in-memory storage as fallback for chat history")
    
    def create_chat(self, title=None, user_id=None):
        """
        Create a new chat session
        
        Args:
            title (str, optional): Chat title
            user_id (str, optional): User identifier
            
        Returns:
            str: Chat ID
        """
        chat_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        chat = {
            'chat_id': chat_id,
            'title': title or f"Chat {timestamp.strftime('%Y-%m-%d %H:%M')}",
            'user_id': user_id,
            'created_at': timestamp,
            'updated_at': timestamp,
            'message_count': 0
        }
        
        try:
            if self.client:
                result = self.chats.insert_one(chat)
                logger.info(f"Created new chat with ID: {chat_id}")
                return chat_id
            else:
                # In-memory fallback
                self.in_memory_chats.append(chat)
                return chat_id
        except Exception as e:
            logger.error(f"Failed to create chat: {str(e)}")
            return None
    
    def get_chat(self, chat_id):
        """
        Get chat by ID
        
        Args:
            chat_id (str): Chat ID
            
        Returns:
            dict: Chat data or None if not found
        """
        try:
            if self.client:
                chat = self.chats.find_one({'chat_id': chat_id})
                return chat
            else:
                # In-memory fallback
                for chat in self.in_memory_chats:
                    if chat['chat_id'] == chat_id:
                        return chat
                return None
        except Exception as e:
            logger.error(f"Failed to get chat {chat_id}: {str(e)}")
            return None
    
    def get_recent_chats(self, limit=10, user_id=None):
        """
        Get recent chat sessions
        
        Args:
            limit (int): Maximum number of chats to return
            user_id (str, optional): Filter by user ID
            
        Returns:
            list: List of chat data
        """
        try:
            if self.client:
                query = {}
                if user_id:
                    query['user_id'] = user_id
                
                chats = list(
                    self.chats.find(
                        query, 
                        {'_id': 0}
                    ).sort('updated_at', -1).limit(limit)
                )
                return chats
            else:
                # In-memory fallback
                sorted_chats = sorted(
                    self.in_memory_chats,
                    key=lambda x: x['updated_at'],
                    reverse=True
                )
                
                if user_id:
                    sorted_chats = [chat for chat in sorted_chats if chat['user_id'] == user_id]
                    
                return sorted_chats[:limit]
        except Exception as e:
            logger.error(f"Failed to get recent chats: {str(e)}")
            return []
    
    def add_message(self, chat_id, role, content, metadata=None):
        """
        Add a message to a chat
        
        Args:
            chat_id (str): Chat ID
            role (str): Message role (user, assistant, system)
            content (str): Message content
            metadata (dict, optional): Additional message metadata
            
        Returns:
            str: Message ID
        """
        if not chat_id or not content:
            logger.error("Chat ID and content are required")
            return None
            
        message_id = str(uuid.uuid4())
        timestamp = datetime.utcnow()
        
        message = {
            'message_id': message_id,
            'chat_id': chat_id,
            'role': role,
            'content': content,
            'metadata': metadata or {},
            'created_at': timestamp
        }
        
        try:
            if self.client:
                # Insert the message
                result = self.messages.insert_one(message)
                
                # Update the chat with the new message count and timestamp
                self.chats.update_one(
                    {'chat_id': chat_id},
                    {
                        '$inc': {'message_count': 1},
                        '$set': {'updated_at': timestamp}
                    }
                )
                
                logger.info(f"Added message {message_id} to chat {chat_id}")
                return message_id
            else:
                # In-memory fallback
                self.in_memory_messages.append(message)
                
                # Update chat
                for chat in self.in_memory_chats:
                    if chat['chat_id'] == chat_id:
                        chat['message_count'] += 1
                        chat['updated_at'] = timestamp
                        break
                
                return message_id
        except Exception as e:
            logger.error(f"Failed to add message to chat {chat_id}: {str(e)}")
            return None
    
    def get_chat_messages(self, chat_id, limit=50, before_id=None):
        """
        Get messages from a chat
        
        Args:
            chat_id (str): Chat ID
            limit (int): Maximum number of messages to return
            before_id (str, optional): Get messages before this message ID
            
        Returns:
            list: List of messages
        """
        try:
            if self.client:
                query = {'chat_id': chat_id}
                
                if before_id:
                    # Get the timestamp of the specified message
                    before_message = self.messages.find_one({'message_id': before_id})
                    if before_message:
                        query['created_at'] = {'$lt': before_message['created_at']}
                
                messages = list(
                    self.messages.find(
                        query, 
                        {'_id': 0}
                    ).sort('created_at', 1).limit(limit)
                )
                return messages
            else:
                # In-memory fallback
                chat_messages = [msg for msg in self.in_memory_messages if msg['chat_id'] == chat_id]
                
                if before_id:
                    before_time = None
                    for msg in chat_messages:
                        if msg['message_id'] == before_id:
                            before_time = msg['created_at']
                            break
                    
                    if before_time:
                        chat_messages = [msg for msg in chat_messages if msg['created_at'] < before_time]
                
                # Sort by creation time
                sorted_messages = sorted(chat_messages, key=lambda x: x['created_at'])
                return sorted_messages[:limit]
        except Exception as e:
            logger.error(f"Failed to get messages for chat {chat_id}: {str(e)}")
            return []
    
    def get_chat_context(self, chat_id, message_limit=10):
        """
        Get chat context for a follow-up query
        
        Args:
            chat_id (str): Chat ID
            message_limit (int): Maximum number of messages to include in context
            
        Returns:
            list: List of messages formatted as context for LLM
        """
        messages = self.get_chat_messages(chat_id, limit=message_limit)
        
        # Format messages for LLM context
        formatted_context = []
        for message in messages:
            formatted_context.append({
                'role': message['role'],
                'content': message['content']
            })
            
        return formatted_context
    
    def update_chat_title(self, chat_id, new_title):
        """
        Update a chat's title
        
        Args:
            chat_id (str): Chat ID
            new_title (str): New chat title
            
        Returns:
            bool: Success status
        """
        try:
            if self.client:
                result = self.chats.update_one(
                    {'chat_id': chat_id},
                    {'$set': {'title': new_title}}
                )
                return result.modified_count > 0
            else:
                # In-memory fallback
                for chat in self.in_memory_chats:
                    if chat['chat_id'] == chat_id:
                        chat['title'] = new_title
                        return True
                return False
        except Exception as e:
            logger.error(f"Failed to update chat title for {chat_id}: {str(e)}")
            return False
    
    def delete_chat(self, chat_id):
        """
        Delete a chat and all its messages
        
        Args:
            chat_id (str): Chat ID
            
        Returns:
            bool: Success status
        """
        try:
            if self.client:
                # Delete the chat
                chat_result = self.chats.delete_one({'chat_id': chat_id})
                
                # Delete all messages for this chat
                message_result = self.messages.delete_many({'chat_id': chat_id})
                
                logger.info(f"Deleted chat {chat_id} with {message_result.deleted_count} messages")
                return chat_result.deleted_count > 0
            else:
                # In-memory fallback
                self.in_memory_chats = [chat for chat in self.in_memory_chats if chat['chat_id'] != chat_id]
                self.in_memory_messages = [msg for msg in self.in_memory_messages if msg['chat_id'] != chat_id]
                return True
        except Exception as e:
            logger.error(f"Failed to delete chat {chat_id}: {str(e)}")
            return False
            
    def generate_chat_title(self, chat_id, llm_service):
        """
        Generate a title for a chat using LLM based on the first message
        
        Args:
            chat_id (str): Chat ID
            llm_service: LLM service instance
            
        Returns:
            str: Generated title
        """
        # Get the first user message
        messages = self.get_chat_messages(chat_id, limit=2)
        user_message = None
        
        for message in messages:
            if message['role'] == 'user':
                user_message = message['content']
                break
        
        if not user_message:
            return None
            
        # Use the template service for title generation
        prompt = PromptTemplateService.TITLE_GENERATION["chat_title"].format(
            message=user_message
        )
        
        try:
            title = llm_service.generate_response(
                prompt=prompt,
                context=None,
                max_tokens=20
            )
            
            # Clean up the title
            title = title.strip().strip('"\'').strip()
            
            # Update the chat with the new title
            if title:
                self.update_chat_title(chat_id, title)
                return title
                
            return None
        except Exception as e:
            logger.error(f"Failed to generate title for chat {chat_id}: {str(e)}")
            return None