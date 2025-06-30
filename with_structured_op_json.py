import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    tools=[
        {
            "function_declarations": [
                {
                    "name": "extract_review_info",
                    "description": "Extract structured review information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key_themes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Key themes in the review"
                            },
                            "summary": {
                                "type": "string",
                                "description": "A brief summary of the review"
                            },
                            "sentiment": {
                                "type": "string",
                                "enum": ["pos", "neg", "neutral"],
                                "description": "Sentiment of the review"
                            },
                            "pros": {
                                "type": ["array", "null"],
                                "items": {"type": "string"},
                                "description": "Pros in the review"
                            },
                            "cons": {
                                "type": ["array", "null"],
                                "items": {"type": "string"},
                                "description": "Cons in the review"
                            },
                            "name": {
                                "type": ["string", "null"],
                                "description": "Name of the reviewer"
                            }
                        },
                        "required": ["key_themes", "summary", "sentiment"]
                    }
                }
            ]
        }
    ]
)

# Prompt
prompt = """The hardware is great, but the software feels bloated. 
There are too many pre-installed apps that I can't remove. 
Also, the UI looks outdated compared to other brands. 
Hoping for a software update to fix this."""

response = model.generate_content(prompt)

# Tool-calling result
print(response.candidates[0].content.parts)
