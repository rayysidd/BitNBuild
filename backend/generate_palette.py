import google.generativeai as genai
import os
from dotenv import load_dotenv # <-- ADD THIS LINE

load_dotenv() # <-- AND ADD THIS LINE to load the .env file

# The script now looks for the 'GOOGLE_API_KEY' variable
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

genai.configure(api_key=API_KEY)

# --- YOUR PROMPT GOES HERE ---
prompt = """
Create a color palette inspired by the movie 'La La Land'.
The mood should be romantic, nostalgic, and vibrant.
Provide a list of 5 colors. For each color, give me its name, its hex code,
and a short description of how it relates to the movie.
"""

# Create the model
model = genai.GenerativeModel('gemini-1.5-flash')

# Send the prompt and get the response
response = model.generate_content(prompt)

# Print the result
print("--- Gemini Color Palette ---")
print(response.text)