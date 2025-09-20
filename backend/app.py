# ai/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import google.generativeai as genai
import json
import re
from dotenv import load_dotenv

# --- Helper modules (ensure these files exist in the same directory) ---
# You will need to have your existing files for these imports.
# If they don't exist, these routes will fail.
try:
    from image_to_palette import extract_palette
    from advanced_ai_palette import optimize_palette, PaletteAestheticNet
except ImportError:
    print("⚠️  Warning: 'image_to_palette' or 'advanced_ai_palette' not found. Image routes will not work.")
    extract_palette = None
    optimize_palette = None
    PaletteAestheticNet = None
# --------------------------------------------------------------------


load_dotenv()
app = Flask(__name__)
CORS(app)

# Configure and initialize the Gemini model at startup
gemini_model = None
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  GEMINI_API_KEY not found in .env file. Text generation API will be disabled.")
    else:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        print("✅ Gemini model initialized successfully.")
except Exception as e:
    print(f"⚠️  Could not initialize Gemini model: {e}")

# Try to load aesthetic model (optional)
aesthetic_model = None
if PaletteAestheticNet:
    try:
        aesthetic_model = PaletteAestheticNet(K=8)
        print("✅ Aesthetic model created (using default weights)")
    except Exception as e:
        print(f"⚠️  Could not create aesthetic model: {e}")

# --- API Routes ---

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to confirm the server is running."""
    return jsonify({"status": "healthy"}), 200

@app.route('/api/generate-from-text', methods=['POST'])
def generate_from_text_api():
    """Generates a color palette from a text prompt using the Gemini API."""
    data = request.get_json()
    user_prompt = data.get('prompt', '')
    
    # --- LOGGING: Print incoming request ---
    print("\n--- [REQUEST RECEIVED] /api/generate-from-text ---")
    print(f"Prompt: {user_prompt}")
    # ----------------------------------------
    
    if not gemini_model:
        return jsonify({"error": "Text generation model is not available. Check API key."}), 503
    if not user_prompt:
        return jsonify({"error": "No 'prompt' key found in the request body"}), 400

    system_prompt = f"""
    You are an expert color palette generator. Based on the user's prompt, create a harmonious and aesthetically pleasing color palette of 6 colors.
    The user's prompt is: "{user_prompt}"

    Your response MUST be a single, valid JSON object. Do not include any text, explanations, or markdown formatting before or after the JSON object.
    The JSON object must have a single key "palette", which is an array of objects.
    Each object in the array must contain:
    1. A "hex" key with the color as a string (e.g., "#RRGGBB").
    2. A "name" key with a creative name for the color as a string.
    """
    
    try:
        response = gemini_model.generate_content(system_prompt)
        text_response = response.text.strip()
        
        # Use regex to find the JSON block, making it robust against markdown fences
        json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
        if not json_match:
            print(f"Error: Gemini returned a non-JSON response: {text_response}")
            return jsonify({"error": "AI model returned an invalid format"}), 500
            
        palette_data = json.loads(json_match.group())

        # --- LOGGING: Print outgoing response ---
        print("--- [RESPONSE SENT] /api/generate-from-text ---")
        print(json.dumps(palette_data, indent=2))
        print("---------------------------------------------")
        # -------------------------------------------

        return jsonify(palette_data)

    except Exception as e:
        print(f"Error during Gemini API call: {str(e)}")
        return jsonify({"error": "Internal server error during AI generation."}), 500


@app.route('/api/extract', methods=['POST'])
def extract_palette_api():
    """Extracts a color palette from an uploaded image."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    optimize_level = request.form.get('optimize', 'basic')

    # --- LOGGING: Print incoming request ---
    print("\n--- [REQUEST RECEIVED] /api/extract ---")
    print(f"Filename: {file.filename}, Optimize Level: {optimize_level}")
    print("---------------------------------------")
    # ----------------------------------------

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if not extract_palette:
         return jsonify({"error": "Image processing module not available on server"}), 500

    try:
        basic_palette = extract_palette(file, num_colors=10)
        if not basic_palette:
            return jsonify({"error": "Could not process the image"}), 500
        
        # For this simplified example, we'll just return the basic palette
        response_data = {
            "palette": basic_palette,
            "message": "Basic color palette extracted successfully",
            "optimization": "basic",
            "colors_found": len(basic_palette)
        }
        
        # --- LOGGING: Print outgoing response ---
        print("--- [RESPONSE SENT] /api/extract ---")
        print(f"Successfully extracted {len(basic_palette)} colors.")
        print("------------------------------------")
        # -------------------------------------------
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({"error": "Internal server error while processing the image."}), 500


if __name__ == '__main__':
    print("🎨 Starting ChromaGen AI Backend...")
    print("📡 Server running at: http://localhost:5000")
    app.run(debug=True, port=5000, host='0.0.0.0')