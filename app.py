import streamlit as st
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import json
from datetime import datetime
import os
import pytesseract
from PIL import Image
import io
import base64
from pdf2image import convert_from_bytes
import openai
from openai import OpenAI

# Set Tesseract Path (Update this path to match your Tesseract installation)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Medical Report Processing Functions
def extract_text_from_image(image):
    """Extract text from image using OCR"""
    try:
        # Try to get Tesseract version to check installation
        try:
            tesseract_version = pytesseract.get_tesseract_version()
            st.success(f"✅ Tesseract OCR version {tesseract_version} found")
        except Exception as e:
            st.error(f"""
            ❌ Tesseract OCR error: {str(e)}
            
            Please ensure Tesseract is installed and properly configured:
            1. Download Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki
            2. Install it in the default location (C:\\Program Files\\Tesseract-OCR)
            3. If installed in a different location, update the path in the code
            4. Restart the application
            
            Current expected path: {pytesseract.pytesseract.tesseract_cmd}
            """)
            return None
            
        # Prepare image for better OCR
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Extract text
        text = pytesseract.image_to_string(image)
        
        if not text or len(text.strip()) < 5:
            st.warning("⚠️ No text was detected in the image. Please ensure the image is clear and contains readable text.")
            return None
            
        return text
    except Exception as e:
        st.error(f"Error in OCR processing: {str(e)}")
        return None

def process_medical_report(file_content, file_type):
    """Process uploaded medical report file"""
    try:
        if file_type.startswith('image/'):
            image = Image.open(io.BytesIO(file_content))
            text = extract_text_from_image(image)
        elif file_type == 'application/pdf':
            images = convert_from_bytes(file_content)
            text = ""
            for image in images:
                text += extract_text_from_image(image) + "\n"
        else:
            text = file_content.decode('utf-8')
        return text
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def interpret_medical_report(text):
    """Interpret medical report using OpenAI API"""
    if not text or len(text.strip()) < 10:
        st.error("Not enough text was extracted from the image. Please ensure the image is clear and contains readable text.")
        return None
        
    prompt = f"""
    Below is a medical report. Please:
    1. Summarize the key findings in simple, non-technical language
    2. Highlight any concerning areas that need attention
    3. Suggest next steps or lifestyle changes based on the results
    4. Indicate if any immediate medical attention is needed

    Medical Report:
    {text}
    """

    try:
        # Get API key from secrets
        api_key = st.secrets.get("OPENAI_API_KEY")
        if not api_key:
            st.error("""
            ⚠️ OpenAI API key not found! Please follow these steps:
            1. Go to https://platform.openai.com/api-keys
            2. Sign in or create an account
            3. Set up billing information
            4. Create a new API key
            5. Add it to .streamlit/secrets.toml as: OPENAI_API_KEY = "your-key-here"
            """)
            return None
            
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=api_key)
        
        # Show a warning about billing setup if needed
        st.info("""
        ℹ️ If you're getting rate limit errors, please ensure you have:
        1. Set up billing information in your OpenAI account
        2. Verified your email address
        3. Added a valid payment method
        
        Visit: https://platform.openai.com/account/billing
        """)
        
        # Make API call with retries
        import time
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful medical assistant that explains medical reports in simple terms. Always remind users to consult healthcare professionals for medical advice."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                break  # If successful, break the retry loop
            except openai.RateLimitError:
                if attempt < max_retries - 1:  # If not the last attempt
                    st.warning(f"Rate limit hit. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise  # Re-raise the error if all retries failed

        # Extract and return the interpretation
        interpretation = response.choices[0].message.content
        return interpretation
        
    except openai.AuthenticationError:
        st.error("Authentication error: Please check your OpenAI API key.")
        return None
    except openai.RateLimitError:
        st.error("Rate limit exceeded: Please try again later.")
        return None
    except openai.APIError as e:
        st.error(f"OpenAI API error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error interpreting report: {str(e)}")
        return None

try:
    # Page configuration
    st.set_page_config(
        page_title="CardioCare AI - Heart Health Assistant",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add tabs for different functionalities
    tab1, tab2 = st.tabs(["Risk Assessment", "Medical Report Analysis"])
    
    with tab2:
        st.header("📋 Medical Report Analysis")
        st.markdown("""
        Upload your medical report or take a picture to get an AI-powered interpretation 
        in simple terms. Supported formats: PDF, Images (JPG, PNG), and Text files.
        """)
        
        # File upload and camera input
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_file = st.file_uploader("Upload Medical Report", 
                type=['pdf', 'png', 'jpg', 'jpeg', 'txt'],
                help="Upload your medical report file")
        
        with col2:
            st.markdown("### 📸 Camera Input")
            camera_enabled = st.button("📸 Turn On Camera", 
                help="Click to enable camera capture",
                use_container_width=True)
            
            if camera_enabled:
                camera_input = st.camera_input("Take a picture of your report",
                    help="Use your camera to take a picture of your medical report")
                if camera_input is not None:
                    st.success("✅ Image captured successfully!")
            else:
                camera_input = None
                st.info("Click the button above to enable camera capture")
        
        if uploaded_file is not None:
            with st.spinner("Processing your medical report..."):
                # Process uploaded file
                file_content = uploaded_file.read()
                file_type = uploaded_file.type
                
                # Extract text from the file
                extracted_text = process_medical_report(file_content, file_type)
                
                if extracted_text:
                    # Get AI interpretation
                    interpretation = interpret_medical_report(extracted_text)
                    
                    if interpretation:
                        st.markdown("### 🤖 AI Interpretation")
                        st.markdown(interpretation)
                        st.info("⚕️ Remember: This AI interpretation is for informational purposes only. Always consult with healthcare professionals for medical advice.")
        
        elif camera_input is not None:
            with st.spinner("Processing your medical report..."):
                # Process camera input
                file_content = camera_input.getvalue()
                file_type = camera_input.type
                
                # Extract text from the image
                extracted_text = process_medical_report(file_content, file_type)
                
                if extracted_text:
                    # Get AI interpretation
                    interpretation = interpret_medical_report(extracted_text)
                    
                    if interpretation:
                        st.markdown("### 🤖 AI Interpretation")
                        st.markdown(interpretation)
                        st.info("⚕️ Remember: This AI interpretation is for informational purposes only. Always consult with healthcare professionals for medical advice.")
    
    # Original content goes in tab1
    with tab1:
        st.write("Page configuration loaded successfully")
except Exception as e:
    st.error(f"Error in page configuration: {str(e)}")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .subtitle {
        font-size: 1.2rem;
        color: #000000;
        text-align: center;
        margin-bottom: 3rem;
        font-style: italic;
        font-weight: 500;
    }
    /* Dark mode text color */
    [data-theme="dark"] {
        --text-color: #ffffff;
    }
    /* Ensure all text has good contrast */
    body {
        color: #000000 !important;
    }
    [data-theme="dark"] body {
        color: #ffffff !important;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .high-risk {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left: 5px solid #f44336;
        color: #3d0000 !important;
    }
    .medium-risk {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left: 5px solid #ff9800;
        color: #3d2200 !important;
    }
    .low-risk {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border-left: 5px solid #4caf50;
        color: #002600 !important;
    }
    .high-risk *, .medium-risk *, .low-risk * {
        color: inherit !important;
    }
    .llm-advice {
        background: rgba(33, 150, 243, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid #2196f3;
        color: var(--text-color, inherit) !important;
    }
    .feature-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        color: #000000 !important;
        margin: 0.5rem 0;
    }
    [data-theme="dark"] .feature-card {
        background: rgba(30, 30, 30, 0.9);
        color: #ffffff !important;
    }
    /* Camera button styling */
    .stButton > button {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border: none;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stButton > button:active {
        transform: translateY(0);
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'heart_disease_model.pkl')
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data['model'], model_data['scaler'], model_data['feature_names']
    except FileNotFoundError:
        st.error("❌ Model file not found. Please ensure 'heart_disease_model.pkl' is in the app directory.")
        return None, None, None

def extract_text_from_image(image):
    """Extract text from image using OCR"""
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error in OCR processing: {str(e)}")
        return None

def process_medical_report(file_content, file_type):
    """Process uploaded medical report file"""
    try:
        if file_type.startswith('image/'):
            image = Image.open(io.BytesIO(file_content))
            text = extract_text_from_image(image)
        elif file_type == 'application/pdf':
            images = convert_from_bytes(file_content)
            text = ""
            for image in images:
                text += extract_text_from_image(image) + "\n"
        else:
            text = file_content.decode('utf-8')
        return text
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def interpret_medical_report(text):
    """Interpret medical report using OpenAI API"""
    try:
        openai.api_key = st.secrets["OPENAI_API_KEY"]
        
        prompt = f"""
        Below is a medical report. Please:
        1. Summarize the key findings in simple, non-technical language
        2. Highlight any concerning areas that need attention
        3. Suggest next steps or lifestyle changes based on the results
        4. Indicate if any immediate medical attention is needed

        Medical Report:
        {text}
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant that explains medical reports in simple terms. Always remind users to consult healthcare professionals for medical advice."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        interpretation = response.choices[0].message.content
        return interpretation
    except Exception as e:
        st.error(f"Error interpreting report: {str(e)}")
        return None

# LLM Integration Functions
class LLMHealthCoach:
    def __init__(self):
        # You can use either OpenAI or Google Gemini
        # Set your API key in Streamlit secrets or environment variables
        self.openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    def generate_health_advice(self, risk_level, risk_probability, patient_data, risk_factors):
        """Generate personalized health advice using LLM"""
        # Create context for the LLM
        context = f"""
        Patient Health Information:
        - Age: {patient_data['age']} years old
        - Risk Assessment: {risk_level} risk level ({risk_probability:.1f}% probability)
        - Blood Pressure: {patient_data['bp']} mm Hg
        - Cholesterol Level: {patient_data['cholesterol']} mg/dl
        - Maximum Heart Rate: {patient_data['heart_rate']} bpm
        - Symptoms: {patient_data['chest_pain']} chest pain, {'with' if patient_data['exercise_angina'] == 'Yes' else 'without'} exercise-induced angina
        
        Key Risk Factors: {', '.join(risk_factors) if risk_factors else 'No major risk factors identified'}
        """
        prompt = f"""
        You are CardioCare AI, a compassionate heart health coach. Based on the patient's cardiovascular risk assessment, provide personalized advice.
        
        {context}
        
        Please provide:
        1. A clear, empathetic explanation of their risk level in simple terms
        2. 3-4 specific, actionable lifestyle recommendations
        3. When to seek medical attention
        4. Encouragement and motivation
        
        Keep the tone warm, professional, and encouraging. Avoid medical jargon.
        """
        # First check if we have any API keys
        if not (self.gemini_api_key or self.openai_api_key):
            return self._fallback_advice(risk_level, risk_factors)
        # Try Gemini first if we have the key
        if self.gemini_api_key:
            try:
                return self._call_gemini(prompt)
            except Exception as e:
                st.warning(f"Gemini API error: {str(e)}. Trying OpenAI as fallback...")
        # Try OpenAI as fallback or primary if no Gemini key
        if self.openai_api_key:
            try:
                return self._call_openai(prompt)
            except Exception as e:
                st.error(f"OpenAI API error: {str(e)}")
        # If all APIs fail or no keys available
        return self._fallback_advice(risk_level, risk_factors)
    
    def _call_gemini(self, prompt):
        """Call Google Gemini API"""
        try:
            base_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.gemini_api_key}"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 800,
                    "topP": 0.8,
                    "topK": 40
                }
            }
            
            response = requests.post(
                f"{base_url}?key={self.gemini_api_key}",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "error" in result:
                    raise Exception(result["error"]["message"])
                if "candidates" in result:
                    text = result["candidates"][0].get("content", {}).get("text", "")
                    if text:
                        return text
                raise Exception("No valid text content in response")
            else:
                error_msg = response.json().get("error", {}).get("message", "Unknown error")
                if "safety" in error_msg.lower():
                    # For safety-related errors, throw an exception to trigger the OpenAI fallback
                    raise Exception("Content safety limits - trying alternative service")
                raise Exception(f"Gemini API error: {error_msg}")
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error: {str(e)}")
        except Exception as e:
            raise Exception(f"Error calling Gemini API: {str(e)}")
    
    def _fallback_advice(self, risk_level, risk_factors):
        """Fallback advice when LLM is not available"""
        if risk_level == "HIGH":
            return """
            🚨 **Your assessment indicates elevated cardiovascular risk.**
            
            **Immediate Actions:**
            • Schedule an appointment with your healthcare provider within the next week
            • Monitor your blood pressure daily if possible
            • Avoid strenuous activities until cleared by a doctor
            
            **Lifestyle Changes:**
            • Reduce sodium intake to less than 2,300mg per day
            • Aim for 30 minutes of gentle walking daily (with doctor approval)
            • Practice stress reduction techniques like deep breathing
            
            **Remember:** This is a screening tool. Professional medical evaluation is essential for proper diagnosis and treatment.
            """
        elif risk_level == "MEDIUM":
            return """
            ⚠️ **Your assessment shows moderate cardiovascular risk.**
            
            **Prevention Focus:**
            • Schedule a routine checkup with your doctor in the next month
            • Start tracking your blood pressure and heart rate
            • Begin incorporating heart-healthy habits now
            
            **Lifestyle Recommendations:**
            • Exercise 150 minutes per week (brisk walking counts!)
            • Follow a Mediterranean-style diet rich in fruits and vegetables
            • Limit processed foods and added sugars
            • Get 7-8 hours of quality sleep nightly
            
            **Good News:** Moderate risk often responds well to lifestyle changes!
            """
        else:
            return """
            ✅ **Great news! Your assessment indicates lower cardiovascular risk.**
            
            **Keep Up The Good Work:**
            • Continue your current healthy habits
            • Get regular checkups as recommended by your doctor
            • Stay active and maintain a balanced diet
            
            **Optimize Your Heart Health:**
            • Aim for 10,000 steps per day
            • Include omega-3 rich foods like fish and nuts
            • Practice mindfulness or meditation for stress management
            • Stay hydrated and limit alcohol consumption
            
            **Prevention is Key:** Maintaining heart health now sets you up for a healthier future!
            """
    
    def generate_weekly_digest(self, user_history):
        """Generate a weekly health digest (placeholder for future enhancement)"""
        return "Weekly digest feature coming soon! Track your progress over time."

# Initialize LLM Health Coach
@st.cache_resource
def get_health_coach():
    return LLMHealthCoach()

def main():
    st.markdown('<h1 class="main-header">🧠❤️ CardioCare AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Dual-Engine Heart Health Assistant: ML Risk Prediction + AI Health Coaching</p>', unsafe_allow_html=True)
    
    # Load model and health coach
    model, scaler, feature_names = load_model()
    health_coach = get_health_coach()
    
    if model is None:
        st.stop()
    
    # Sidebar for inputs
    st.sidebar.header("🩺 Patient Information")
    st.sidebar.markdown("*Please provide accurate information for best results*")
    
    # Collect input features
    age = st.sidebar.slider("Age", 20, 100, 50)
    
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    sex_numeric = 1 if sex == "Male" else 0
    
    cp = st.sidebar.selectbox(
        "Chest Pain Type",
        ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"],
        help="Type of chest pain experienced"
    )
    cp_mapping = {
        "Typical Angina": 0, "Atypical Angina": 1, 
        "Non-anginal Pain": 2, "Asymptomatic": 3
    }
    cp_numeric = cp_mapping[cp]
    
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 220, 120)
    chol = st.sidebar.slider("Cholesterol Level (mg/dl)", 100, 600, 240)
    
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    fbs_numeric = 1 if fbs == "Yes" else 0
    
    restecg = st.sidebar.selectbox(
        "Resting ECG Results",
        ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"]
    )
    restecg_mapping = {
        "Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2
    }
    restecg_numeric = restecg_mapping[restecg]
    
    thalch = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)
    
    exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
    exang_numeric = 1 if exang == "Yes" else 0
    
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", 0.0, 6.0, 1.0, 0.1)
    
    slope = st.sidebar.selectbox(
        "Slope of Peak Exercise ST Segment",
        ["Upsloping", "Flat", "Downsloping"]
    )
    slope_mapping = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    slope_numeric = slope_mapping[slope]
    
    ca = st.sidebar.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 4, 0)
    
    thal = st.sidebar.selectbox(
        "Thalassemia", ["Normal", "Fixed Defect", "Reversable Defect"]
    )
    thal_mapping = {"Normal": 0, "Fixed Defect": 1, "Reversable Defect": 2}
    thal_numeric = thal_mapping[thal]
    
    # Create feature array
    features = np.array([[
        age, sex_numeric, cp_numeric, trestbps, chol, fbs_numeric,
        restecg_numeric, thalch, exang_numeric, oldpeak, 
        slope_numeric, ca, thal_numeric
    ]])
    
    # Main content area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.header("🔍 ML Risk Assessment")
        
        if st.button("🚀 Analyze Heart Health Risk", type="primary", use_container_width=True):
            with st.spinner("🧠 ML Model analyzing your data..."):
                # Scale features and make prediction
                features_scaled = scaler.transform(features)
                prediction = model.predict(features_scaled)[0]
                probability = model.predict_proba(features_scaled)[0]
                risk_probability = probability[1] * 100
                
                # Determine risk level
                if risk_probability >= 70:
                    risk_level = "HIGH"
                    risk_class = "high-risk"
                    risk_icon = "🚨"
                elif risk_probability >= 30:
                    risk_level = "MEDIUM"
                    risk_class = "medium-risk"
                    risk_icon = "⚠️"
                else:
                    risk_level = "LOW"
                    risk_class = "low-risk"
                    risk_icon = "✅"
                
                # Display ML prediction results
                st.markdown(f"""
                <div class="prediction-box {risk_class}">
                    <h3>{risk_icon} {risk_level} RISK</h3>
                    <p><strong>Risk Probability: {risk_probability:.1f}%</strong></p>
                    <p><em>Based on machine learning analysis of your health parameters</em></p>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk probability gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk_probability,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Cardiovascular Risk Score"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#f44336" if risk_probability >= 70 else "#ff9800" if risk_probability >= 30 else "#4caf50"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "#ffeb3b"},
                            {'range': [70, 100], 'color': "#ffcdd2"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # Identify risk factors
                risk_factors = []
                if age > 60:
                    risk_factors.append(f"Advanced age ({age} years)")
                if sex == "Male":
                    risk_factors.append("Male gender")
                if trestbps > 140:
                    risk_factors.append(f"High blood pressure ({trestbps} mm Hg)")
                if chol > 240:
                    risk_factors.append(f"High cholesterol ({chol} mg/dl)")
                if fbs == "Yes":
                    risk_factors.append("Elevated fasting blood sugar")
                if thalch < 100:
                    risk_factors.append(f"Low maximum heart rate ({thalch} bpm)")
                if exang == "Yes":
                    risk_factors.append("Exercise induced angina")
                if oldpeak > 2:
                    risk_factors.append(f"Significant ST depression ({oldpeak})")
                if cp == "Asymptomatic":
                    risk_factors.append("Asymptomatic chest pain")
                
                # Prepare patient data for LLM
                patient_data = {
                    'age': age,
                    'gender': sex,
                    'bp': trestbps,
                    'cholesterol': chol,
                    'heart_rate': thalch,
                    'exercise_angina': exang,
                    'chest_pain': cp
                }
                
                # LLM Health Coaching Section
                st.header("🤖 AI Health Coach")
                
                with st.spinner("🧠 AI Health Coach analyzing your results..."):
                    llm_advice = health_coach.generate_health_advice(
                        risk_level, risk_probability, patient_data, risk_factors
                    )
                
                st.markdown(f"""
                <div class="llm-advice">
                    <h4>💬 Personalized Health Guidance</h4>
                    <div style="white-space: pre-line;">{llm_advice}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk factors analysis
                st.subheader("📊 Risk Factor Analysis")
                
                if risk_factors:
                    st.warning(f"⚠️ **{len(risk_factors)} Risk Factor(s) Identified:**")
                    for i, factor in enumerate(risk_factors, 1):
                        st.markdown(f"""
                        <div class="feature-card">
                            <strong>{i}.</strong> {factor}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ **No major risk factors identified based on standard thresholds**")
                
                # Additional insights
                st.subheader("🔬 Model Insights")
                
                # Feature importance (simplified)
                feature_impact = {
                    'Chest Pain Type': abs(cp_numeric - 1.5) * 20,
                    'Age Factor': (age - 40) * 0.5 if age > 40 else 0,
                    'Blood Pressure': max(0, (trestbps - 120) * 0.3),
                    'Cholesterol Impact': max(0, (chol - 200) * 0.1),
                    'Heart Rate Factor': max(0, (150 - thalch) * 0.2)
                }
                
                impact_df = pd.DataFrame(
                    list(feature_impact.items()), 
                    columns=['Factor', 'Impact Score']
                ).sort_values('Impact Score', ascending=True)
                
                fig_impact = px.bar(
                    impact_df, 
                    x='Impact Score', 
                    y='Factor',
                    orientation='h',
                    title="Key Factors Contributing to Your Risk Score",
                    color='Impact Score',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_impact, use_container_width=True)
    
    with col2:
        st.header("📋 Patient Summary")
        
        summary_data = {
            "Parameter": [
                "Age", "Sex", "Chest Pain", "Blood Pressure",
                "Cholesterol", "Fasting Blood Sugar", "Resting ECG",
                "Max Heart Rate", "Exercise Angina", "ST Depression",
                "ST Slope", "Major Vessels", "Thalassemia"
            ],
            "Value": [
                f"{age} years", sex, cp, f"{trestbps} mm Hg",
                f"{chol} mg/dl", fbs, restecg, f"{thalch} bpm",
                exang, f"{oldpeak}", slope, ca, thal
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Quick Health Tips
        st.header("💡 Daily Heart Health Tips")
        tips = [
            "🚶‍♂️ Walk 30 minutes daily",
            "🥗 Eat 5 servings of fruits/vegetables",
            "💧 Drink 8 glasses of water",
            "😴 Get 7-8 hours of sleep",
            "🧘‍♀️ Practice 10 minutes of meditation",
            "🚭 Avoid smoking and secondhand smoke"
        ]
        
        for tip in tips:
            st.markdown(f"• {tip}")
        
        # API Status
        st.header("🔧 System Status")
        health_coach = get_health_coach()
        if health_coach.openai_api_key:
            st.success("✅ OpenAI GPT-4 Connected")
        elif health_coach.gemini_api_key:
            st.success("✅ Google Gemini Connected") 
        else:
            st.warning("⚠️ Using Fallback AI Coach")
        
        st.info("💡 **Tip**: Add your API key in Streamlit secrets for enhanced AI coaching!")

# About section
def show_about():
    st.header("ℹ️ About CardioCare AI")
    
    st.markdown("""
    ### 🎯 Dual-Engine Architecture
    
    **Engine 1: Machine Learning Risk Predictor**
    - Trained on cardiovascular health datasets
    - Uses advanced algorithms (Random Forest/Logistic Regression)
    - Provides quantitative risk assessment
    
    **Engine 2: LLM Health Coach**
    - Powered by GPT-4 or Google Gemini
    - Explains results in plain language
    - Provides personalized lifestyle recommendations
    - Offers encouragement and motivation
    
    ### 🔬 Technical Features
    - **Real-time Analysis**: Instant risk scoring
    - **Personalized Coaching**: AI-generated advice
    - **Visual Insights**: Interactive charts and gauges  
    - **Risk Factor Analysis**: Detailed health parameter breakdown
    - **Secure Processing**: No personal data stored
    
    ### ⚠️ Important Disclaimer
    This application is designed for educational and screening purposes only. 
    It should **never replace professional medical advice, diagnosis, or treatment**. 
    Always consult with qualified healthcare professionals for medical decisions.
    
    ### 🛠️ Technical Stack
    - **ML Framework**: Scikit-learn
    - **LLM Integration**: OpenAI GPT-4 / Google Gemini
    - **Frontend**: Streamlit
    - **Visualization**: Plotly
    - **Data Processing**: Pandas, NumPy
    """)

# Main app navigation
def main_nav():
    st.sidebar.markdown("---")
    
    # Navigation
    page = st.sidebar.selectbox(
        "📍 Navigation",
        ["🏠 Main Assessment", "ℹ️ About", "📊 Weekly Digest"],
        index=0
    )
    
    if page == "🏠 Main Assessment":
        main()
    elif page == "ℹ️ About":
        show_about()
    elif page == "📊 Weekly Digest":
        st.header("📅 Weekly Health Digest")
        st.info("🚧 Feature coming soon! Track your health progress over time.")
        
        # Placeholder for weekly digest
        sample_digest = """
        ### Your Weekly Heart Health Summary
        
        **This Week's Highlights:**
        - 3 risk assessments completed
        - Average risk score: 25% (Low Risk)
        - Key improvement areas: Exercise frequency
        
        **AI Coach Recommendations:**
        - Continue your current exercise routine
        - Consider adding strength training 2x/week
        - Monitor blood pressure weekly
        
        **Next Week's Goals:**
        - Walk 10,000 steps daily
        - Reduce sodium intake by 500mg
        - Practice stress reduction techniques
        """
        
        st.markdown(sample_digest)

# Run the app
if __name__ == "__main__":
    main_nav()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d;">
    <p>🧠❤️ <strong>CardioCare AI</strong> - Dual-Engine Heart Health Assistant</p>
    <p><em>ML Risk Prediction + AI Health Coaching | Educational Use Only</em></p>
</div>
""", unsafe_allow_html=True)