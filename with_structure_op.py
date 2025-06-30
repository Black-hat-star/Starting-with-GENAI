from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal, Optional
import os

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini model with API key
model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", api_key=api_key)

# Define the structured output schema
class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["pos", "neg", "neutral"] = Field(description="Return sentiment of the review either positive, negative, or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside a list")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")

# Convert the model to use structured output
structured_model = model.with_structured_output(Review)

# Input text (with no name for the reviewer)
text = """The hardware is great, but the software feels bloated. 
There are too many pre-installed apps that I can't remove. 
Also, the UI looks outdated compared to other brands. 
Hoping for a software update to fix this."""

# Run the model
result = structured_model.invoke(text)

# Print structured result
print("----- Extracted Review -----")
print(f"Summary   : {result.summary}")
print(f"Sentiment : {result.sentiment}")
print(f"Key Themes: {result.key_themes}")
print(f"Pros      : {result.pros or []}")
print(f"Cons      : {result.cons or []}")
print(f"Reviewer  : {result.name or 'Anonymous'}")  # fallback if name is None
