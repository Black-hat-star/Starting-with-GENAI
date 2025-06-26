from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
import os

load_dotenv()

# Initialize Gemini model
model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash")

chat_history =[
    SystemMessage(content='You are a helpful ai assistant')
]
while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.lower() in ['exit', 'quit']:
        print("Chatbot: Goodbye!")
        break

    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content=response.content))
    print("Chatbot:", response.content)

print(chat_history)
