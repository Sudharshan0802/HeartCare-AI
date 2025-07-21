import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from groq import Groq
from dotenv import load_dotenv
import os
import re
import time

load_dotenv()

# Configuration
REQUIRED_FIELDS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalch", "exang", "oldpeak", "slope", "ca", "thal"
]

FIELD_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (0: Female, 1: Male)",
    "cp": "Chest pain type (0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Cholesterol level (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (0: No, 1: Yes)",
    "restecg": "Resting ECG results (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy)",
    "thalch": "Maximum heart rate achieved",
    "exang": "Exercise induced angina (0: No, 1: Yes)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of peak exercise ST segment (0: Upsloping, 1: Flat, 2: Downsloping)",
    "ca": "Number of major vessels colored by fluoroscopy (0-3)",
    "thal": "Thalassemia (1: Normal, 2: Fixed defect, 3: Reversible defect)"
}

VALID_RANGES = {
    "age": (20, 100), "sex": (0, 1), "cp": (0, 3), "trestbps": (90, 200),
    "chol": (100, 600), "fbs": (0, 1), "restecg": (0, 2), "thalch": (60, 220),
    "exang": (0, 1), "oldpeak": (0, 10), "slope": (0, 2), "ca": (0, 3), "thal": (1, 3)
}

# UI Setup
st.set_page_config(page_title="HeartCare AI", page_icon="❤️", layout="centered")
st.title("❤️ HeartCare AI Assistant")
st.markdown("*AI-powered heart disease risk assessment through conversational interface*")

# Load ML Model
@st.cache_resource
def load_prediction_model():
    """Load the trained model and scaler"""
    try:
        model = tf.keras.models.load_model("models/best_global_fl_model.h5")
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, scaler = load_prediction_model()

def predict_heart_disease(patient_data):
    """Make heart disease prediction using the global FL model"""
    if model is None or scaler is None:
        return {"error": "Model not loaded"}
    
    # Prepare features in the correct order
    features = np.array([
        patient_data['age'], patient_data['sex'], patient_data['cp'], 
        patient_data['trestbps'], patient_data['chol'], patient_data['fbs'],
        patient_data['restecg'], patient_data['thalch'], patient_data['exang'],
        patient_data['oldpeak'], patient_data['slope'], patient_data['ca'], 
        patient_data['thal']
    ]).astype(np.float32).reshape(1, -1)
    
    # Standardize specific features (age, trestbps, chol, thalch, oldpeak)
    features_to_scale = features[:, [0, 3, 4, 7, 9]].copy()
    scaled_features = scaler.transform(features_to_scale)
    features[:, [0, 3, 4, 7, 9]] = scaled_features
    
    # Make prediction
    probability = model.predict(features, verbose=0)[0][0]
    
    return {
        'probability': float(probability),
        'risk_class': 'High Risk' if probability > 0.5 else 'Low Risk',
        'confidence': float(probability) if probability > 0.5 else float(1 - probability)
    }

def extract_medical_data_with_llm(user_input):
    """Use LLM to extract medical data from user input - much more accurate than regex"""
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        extraction_prompt = f"""Extract medical information from the following text and return ONLY a JSON object with the extracted values. If a piece of information is not mentioned, omit that field from the JSON.

Medical fields to extract:
- age: Patient's age (20-100)
- sex: 0 for female, 1 for male
- cp: Chest pain type (0=typical angina, 1=atypical angina, 2=non-anginal pain, 3=asymptomatic/no chest pain)
- trestbps: Resting blood pressure (90-200)
- chol: Cholesterol level (100-600)
- fbs: Fasting blood sugar >120 mg/dl (0=no, 1=yes)
- restecg: Resting ECG (0=normal, 1=ST-T abnormality, 2=left ventricular hypertrophy)
- thalch: Maximum heart rate achieved (60-220)
- exang: Exercise induced angina (0=no, 1=yes)
- oldpeak: ST depression induced by exercise (0-10, can be decimal)
- slope: Slope of peak exercise ST segment (0=upsloping, 1=flat, 2=downsloping)
- ca: Number of major vessels blocked (0-3)
- thal: Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)

Patient input: "{user_input}"

Return only valid JSON with extracted values:"""

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": extraction_prompt}],
            temperature=0.1,
            max_tokens=200
        )
        
        # Parse the JSON response
        import json
        extracted_text = response.choices[0].message.content.strip()
        
        # Clean up the response - sometimes LLM adds extra text
        if extracted_text.startswith("```json"):
            extracted_text = extracted_text.replace("```json", "").replace("```", "").strip()
        elif extracted_text.startswith("```"):
            extracted_text = extracted_text.replace("```", "").strip()
        
        # Find JSON object in the response
        start_idx = extracted_text.find('{')
        end_idx = extracted_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx != 0:
            json_str = extracted_text[start_idx:end_idx]
            extracted_data = json.loads(json_str)
            
            # Validate extracted data
            validated_data = {}
            for field, value in extracted_data.items():
                if field in REQUIRED_FIELDS and value is not None:
                    # Validate ranges
                    min_val, max_val = VALID_RANGES.get(field, (float('-inf'), float('inf')))
                    try:
                        num_value = float(value)
                        if min_val <= num_value <= max_val:
                            validated_data[field] = int(num_value) if num_value.is_integer() else num_value
                    except (ValueError, AttributeError):
                        pass
            
            return validated_data
        else:
            return {}
            
    except Exception as e:
        print(f"LLM extraction error: {e}")
        # Fallback to basic regex for critical fields
        return extract_basic_data(user_input)

def extract_basic_data(user_input):
    """Fallback basic extraction for critical fields"""
    extracted = {}
    text = user_input.lower()
    
    # Age
    age_match = re.search(r'(\d+)\s*[-\s]*years?\s*old|i\'?m\s*(\d+)', text)
    if age_match:
        age = int(age_match.group(1) or age_match.group(2))
        if 20 <= age <= 100:
            extracted['age'] = age
    
    # Sex
    if any(word in text for word in ['male', 'man']):
        extracted['sex'] = 1
    elif any(word in text for word in ['female', 'woman']):
        extracted['sex'] = 0
    
    return extracted

def get_next_question():
    """Determine what information to ask for next"""
    missing = []
    for field in REQUIRED_FIELDS:
        if field not in st.session_state.patient_data or st.session_state.patient_data[field] is None:
            missing.append(field)
    
    if not missing:
        return None
    
    # Prioritize clinically important fields
    priority_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalch', 'exang']
    for field in priority_fields:
        if field in missing:
            return field
    
    return missing[0]

def generate_question(field):
    """Generate a natural question for the given field"""
    questions = {
        'age': "What's your age?",
        'sex': "Are you male or female? (Please specify: male/female)",
        'cp': "Do you experience chest pain? If yes, what type? (typical angina, atypical angina, non-anginal pain, or no chest pain)",
        'trestbps': "What's your resting blood pressure? (e.g., 120)",
        'chol': "What's your cholesterol level? (in mg/dl)",
        'fbs': "Is your fasting blood sugar greater than 120 mg/dl? (yes/no)",
        'restecg': "Do you have any resting ECG abnormalities? (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy)",
        'thalch': "What's your maximum heart rate during exercise?",
        'exang': "Do you experience chest pain during exercise? (yes/no)",
        'oldpeak': "Do you have ST depression during exercise? (numeric value, usually 0-6)",
        'slope': "What's the slope of your peak exercise ST segment? (0: Upsloping, 1: Flat, 2: Downsloping)",
        'ca': "How many major vessels are blocked? (0-3)",
        'thal': "Do you have thalassemia? (1: Normal, 2: Fixed defect, 3: Reversible defect)"
    }
    return questions.get(field, f"Please provide information about {field}")

def get_llm_response(user_input, context):
    """Get response from LLM based on current context"""
    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        prompt = f"""You are a medical assistant helping collect information for heart disease risk assessment.
        
Context: {context}
User input: {user_input}

Guidelines:
1. Be conversational and empathetic
2. Ask one question at a time
3. Acknowledge any information provided
4. For emergency symptoms, recommend immediate medical attention
5. Keep responses concise and clear
6. Don't provide medical diagnosis, only risk assessment

Respond naturally and helpfully."""

        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"I'm having trouble processing your request. Could you please try again?"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your HeartCare AI assistant. I'll help assess your heart disease risk by asking about your health information. Let's start - what's your age?"}
    ]

if "patient_data" not in st.session_state:
    st.session_state.patient_data = {}

if "assessment_complete" not in st.session_state:
    st.session_state.assessment_complete = False

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_input := st.chat_input("Type your response here..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Check for emergency keywords
    emergency_keywords = ["severe chest pain", "crushing pain", "can't breathe", "heart attack"]
    if any(keyword in user_input.lower() for keyword in emergency_keywords):
        emergency_response = "🚨 **EMERGENCY**: These symptoms require immediate medical attention. Please call emergency services or go to the nearest hospital right away!"
        st.session_state.messages.append({"role": "assistant", "content": emergency_response})
        with st.chat_message("assistant"):
            st.markdown(emergency_response)
        st.stop()
    
    # Extract medical data from input using LLM
    extracted_data = extract_medical_data_with_llm(user_input)
    st.session_state.patient_data.update(extracted_data)
    
    # Check if assessment is complete
    missing_fields = [f for f in REQUIRED_FIELDS if f not in st.session_state.patient_data]
    
    if not missing_fields and not st.session_state.assessment_complete:
        # Make prediction
        with st.spinner("Analyzing your heart disease risk..."):
            time.sleep(2)
            prediction = predict_heart_disease(st.session_state.patient_data)
            
            if "error" not in prediction:
                risk_prob = prediction['probability']
                risk_class = prediction['risk_class']
                
                # Create detailed response
                if risk_class == "High Risk":
                    response = f"""🔴 **Heart Disease Risk Assessment Complete**

Based on your provided information, your heart disease risk is **{risk_prob:.1%}** - classified as **{risk_class}**.

**Recommendations:**
- Consult with a cardiologist as soon as possible
- Consider lifestyle modifications (diet, exercise, stress management)
- Regular monitoring of heart health indicators
- Follow up with your healthcare provider

*Note: This is a risk assessment tool and not a medical diagnosis. Please consult healthcare professionals for proper medical advice.*"""
                else:
                    response = f"""🟢 **Heart Disease Risk Assessment Complete**

Great news! Your heart disease risk is **{risk_prob:.1%}** - classified as **{risk_class}**.

**Recommendations:**
- Maintain your current healthy lifestyle
- Continue regular health checkups
- Keep monitoring your heart health indicators
- Stay active and eat a balanced diet

*Note: This assessment is based on the information provided. Continue regular medical checkups for optimal heart health.*"""
                
                st.session_state.assessment_complete = True
            else:
                response = "I encountered an issue with the risk assessment. Please ensure all information is provided correctly."
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
    
    else:
        # Continue collecting information
        next_field = get_next_question()
        if next_field:
            # Acknowledge extracted data and ask next question
            acknowledgment = ""
            if extracted_data:
                extracted_info = ", ".join([f"{k}: {v}" for k, v in extracted_data.items()])
                acknowledgment = f"Thank you! I've noted: {extracted_info}. "
            
            question = generate_question(next_field)
            context = f"Currently collecting: {next_field}. Already have: {list(st.session_state.patient_data.keys())}"
            
            # Get LLM response
            llm_response = get_llm_response(user_input, context)
            full_response = acknowledgment + llm_response
            
            # If LLM response doesn't include a question, add the specific question
            if "?" not in llm_response:
                full_response += f" {question}"
        else:
            full_response = "Thank you for providing all the information!"
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        with st.chat_message("assistant"):
            st.markdown(full_response)

# Sidebar with progress and collected data
with st.sidebar:
    st.header("Assessment Progress")
    
    collected = len([f for f in REQUIRED_FIELDS if f in st.session_state.patient_data])
    total = len(REQUIRED_FIELDS)
    progress = collected / total
    
    st.progress(progress)
    st.write(f"Collected: {collected}/{total} fields")
    
    if st.session_state.patient_data:
        st.subheader("Collected Information")
        for field, value in st.session_state.patient_data.items():
            if value is not None:
                st.write(f"**{field.upper()}**: {value}")
    
    if st.button("Reset Assessment"):
        st.session_state.clear()
        st.rerun()

# Display model status
if model is None:
    st.error("⚠️ Model not loaded. Please ensure the model files are in the correct location.")
else:
    st.success("✅ Heart disease prediction model loaded successfully")