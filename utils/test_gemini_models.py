"""
Test script to find available Gemini models
Run this with: streamlit run test_gemini_models.py
"""
import streamlit as st
import google.generativeai as genai

st.title("🔍 Gemini Model Checker")

# Get API key from secrets
try:
    if 'GEMINI_API_KEY' in st.secrets:
        api_key = st.secrets['GEMINI_API_KEY']
        genai.configure(api_key=api_key)
        st.success("✅ API Key loaded successfully!")
        
        st.subheader("Available Models:")
        
        try:
            models = genai.list_models()
            
            st.write("### Models that support generateContent:")
            for model in models:
                if 'generateContent' in model.supported_generation_methods:
                    st.write(f"**Model Name:** `{model.name}`")
                    st.write(f"- Display Name: {model.display_name}")
                    st.write(f"- Supported Methods: {', '.join(model.supported_generation_methods)}")
                    st.write("---")
            
            st.write("### All Available Models:")
            for model in models:
                st.write(f"- `{model.name}` - {model.display_name}")
                
        except Exception as e:
            st.error(f"Error listing models: {e}")
            st.write("Full error details:")
            st.exception(e)
            
    else:
        st.error("❌ GEMINI_API_KEY not found in secrets")
        st.info("Add your API key to .streamlit/secrets.toml")
        
except Exception as e:
    st.error(f"Configuration error: {e}")
    st.exception(e)

st.markdown("---")
st.subheader("Test a Model")

model_name = st.text_input("Enter model name to test:", "gemini-pro")
test_prompt = st.text_area("Test prompt:", "Hello! Say hi back.")

if st.button("Test Model"):
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(test_prompt)
        st.success("✅ Model works!")
        st.write("**Response:**")
        st.write(response.text)
    except Exception as e:
        st.error(f"❌ Model test failed: {e}")
        st.exception(e)