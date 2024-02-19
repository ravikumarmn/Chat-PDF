from langchain.pydantic_v1 import BaseModel
import uuid

class User(BaseModel):
    name: str = "User"

    def create_new_user(self):
        user_id = self.generate_id()
        conversation_id = self.generate_id()
        return user_id, conversation_id
    
    @staticmethod
    def generate_id():
        return f"{uuid.uuid4()}"