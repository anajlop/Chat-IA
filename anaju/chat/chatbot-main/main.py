from fastapi import FastAPI, Form
from pydantic import BaseModel
from chatbot.chatbot import respuesta

app = FastAPI()

class Message(BaseModel):
    message: str

@app.post("/chatbot/")
def chatbot_api(message: Message):
    user_message = message.message
    bot_response = respuesta(user_message)
    return {"bot_response": bot_response}
