# test_gemini_api.py
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables from a .env file in the same directory as this script
# Create a .env file next to this script with your GOOGLE_API_KEY
load_dotenv() 

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("🔴 ERROR: GOOGLE_API_KEY not found in environment variables.")
    print("Please create a .env file in the same directory as this script with:")
    print("GOOGLE_API_KEY=your_actual_api_key_here")
    exit()

print(f"🔑 Attempting to use GOOGLE_API_KEY (last 4 chars): ...{api_key[-4:]}")

try:
    genai.configure(api_key=api_key)

    # Use a model that is typically available via API key, e.g., gemini-1.5-flash or gemini-pro
    # Check Google AI Studio for available model names if unsure.
    model = genai.GenerativeModel('gemini-1.5-flash-latest') # Or 'gemini-pro'

    print("\n💬 Sending a simple prompt to the Gemini API...")
    response = model.generate_content("Tell me a short joke.")
    
    print("\n✅ API Call Successful!")
    print("Response from Gemini:")
    print(response.text)

except Exception as e:
    print(f"\n🔴 ERROR during Gemini API call: {type(e).__name__} - {e}")
    if "quota" in str(e).lower():
        print("\n🆘 This is likely a QUOTA issue with the Google AI Studio Gemini API for the provided key.")
        print("   - Check your usage and limits at https://aistudio.google.com/ (under 'API Keys' or similar section).")
        print("   - RPM (Requests Per Minute) for the free tier is a common limit to hit.")
        print("   - Try waiting for a few minutes (5-10 min) for the RPM quota to reset.")
    elif "permission" in str(e).lower() or "denied" in str(e).lower():
        print("\n🆘 This could be a PERMISSION issue.")
        print("   - Ensure the API key is valid and enabled for the Gemini API.")
        print("   - Ensure the Google account owning the key has access to the Gemini API.")
    elif "API key not valid" in str(e):
        print("\n🆘 The API key is not valid. Please check the key.")