# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import requests
import json
from io import BytesIO
import base64

# Try to import the required libraries, with fallbacks
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    st.warning("DeepFace not available. Facial emotion detection will be limited.")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    st.warning("TextBlob not available. Text emotion analysis will be limited.")

# Set page configuration
st.set_page_config(
    page_title="AI MoodCare - Your Mental Wellness Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("🧠 AI MoodCare - Emotion & Mental Wellness Assistant")
st.markdown("""
Welcome to your personal AI-powered mental wellness assistant. 
This app helps you track your emotions, provides personalized support, 
and offers resources to improve your mental wellbeing.
""")

# Initialize session state for mood logging
if 'mood_log' not in st.session_state:
    st.session_state.mood_log = []
if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = "Neutral"

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a feature", 
    ["Home", "Facial Emotion Detection", "Text Emotion Analysis", "Mood Logger", "Wellness Resources"])

# Home page
if app_mode == "Home":
    st.header("Welcome to AI MoodCare")
    st.markdown("""
    ### How it works:
    
    1. **Facial Emotion Detection**: Use your camera to detect your current emotional state
    2. **Text Emotion Analysis**: Share your thoughts and get emotional insights
    3. **Mood Logger**: Track your daily moods and patterns
    4. **Wellness Resources**: Access personalized mental wellness resources
    
    ### Why track your emotions?
    
    - Increases emotional awareness
    - Helps identify patterns and triggers
    - Provides insights for personal growth
    - Supports mental wellness journey
    """)
    
    # Display current mood if available
    if st.session_state.mood_log:
        st.subheader("Your Recent Moods")
        recent_moods = st.session_state.mood_log[-5:]  # Last 5 entries
        for mood in reversed(recent_moods):
            st.write(f"{mood['date']}: {mood['emotion']} - {mood['notes']}")

# Facial Emotion Detection
elif app_mode == "Facial Emotion Detection":
    st.header("Facial Emotion Detection")
    
    if not DEEPFACE_AVAILABLE:
        st.error("""
        DeepFace is not available. Please install it using:
        `pip install deepface`
        """)
        st.info("For now, you can use the Text Emotion Analysis or Mood Logger features.")
    else:
        st.markdown("""
        Allow camera access to detect your emotions in real-time.
        The AI will analyze your facial expressions and provide appropriate wellness support.
        """)
        
        # Camera input
        img_file_buffer = st.camera_input("Take a picture for emotion analysis")
        
        if img_file_buffer is not None:
            # Convert the image to OpenCV format
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Save the image temporarily
            with open("temp_image.jpg", "wb") as f:
                f.write(bytes_data)
            
            # Analyze emotion
            try:
                analysis = DeepFace.analyze(img_path="temp_image.jpg", actions=['emotion'])
                dominant_emotion = analysis[0]['dominant_emotion']
                emotion_scores = analysis[0]['emotion']
                
                # Store the detected emotion
                st.session_state.current_emotion = dominant_emotion
                
                # Display results
                st.subheader(f"Detected Emotion: {dominant_emotion}")
                
                # Show emotion scores as a bar chart
                emotion_df = pd.DataFrame.from_dict(emotion_scores, orient='index', columns=['Score'])
                emotion_df = emotion_df.reset_index()
                emotion_df.columns = ['Emotion', 'Score']
                st.bar_chart(emotion_df.set_index('Emotion'))
                
                # Provide emotion-specific support
                st.subheader("Personalized Support")
                provide_emotion_support(dominant_emotion)
                
            except Exception as e:
                st.error(f"Error in emotion detection: {str(e)}")
                st.info("Please try again with a clearer face image.")

# Text Emotion Analysis
elif app_mode == "Text Emotion Analysis":
    st.header("Text Emotion Analysis")
    
    if not TEXTBLOB_AVAILABLE:
        st.error("""
        TextBlob is not available. Please install it using:
        `pip install textblob`
        """)
        st.info("For now, you can use the Facial Emotion Detection or Mood Logger features.")
    else:
        st.markdown("""
        Share your thoughts, feelings, or anything on your mind. 
        The AI will analyze the emotional tone and provide supportive responses.
        """)
        
        user_text = st.text_area("How are you feeling today?", height=150)
        
        if st.button("Analyze Emotion") and user_text:
            # Analyze sentiment with TextBlob
            blob = TextBlob(user_text)
            sentiment = blob.sentiment
            
            # Determine emotion based on polarity and subjectivity
            polarity = sentiment.polarity
            subjectivity = sentiment.subjectivity
            
            if polarity > 0.5:
                emotion = "Very Positive"
            elif polarity > 0.1:
                emotion = "Positive"
            elif polarity < -0.5:
                emotion = "Very Negative"
            elif polarity < -0.1:
                emotion = "Negative"
            else:
                emotion = "Neutral"
            
            # Store the detected emotion
            st.session_state.current_emotion = emotion
            
            # Display results
            st.subheader("Analysis Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Emotion", emotion)
                st.metric("Polarity", f"{polarity:.2f}")
            
            with col2:
                st.metric("Subjectivity", f"{subjectivity:.2f}")
                st.metric("Confidence", "High" if abs(polarity) > 0.3 else "Medium")
            
            # Provide text-based support
            st.subheader("Personalized Support")
            provide_text_support(emotion, polarity, user_text)

# Mood Logger
elif app_mode == "Mood Logger":
    st.header("Mood Logger")
    
    st.markdown("""
    Track your daily moods to identify patterns and better understand your emotional wellbeing.
    """)
    
    # Mood input form
    with st.form("mood_form"):
        st.subheader("Log Your Current Mood")
        
        # Use detected emotion or allow manual selection
        detected_emotion = st.session_state.current_emotion
        emotion_options = ["Happy", "Sad", "Angry", "Fear", "Surprise", "Neutral", "Anxious", "Calm", "Excited", "Tired"]
        
        if detected_emotion in emotion_options:
            default_index = emotion_options.index(detected_emotion)
        else:
            default_index = emotion_options.index("Neutral")
            
        selected_emotion = st.selectbox("How are you feeling?", emotion_options, index=default_index)
        mood_notes = st.text_area("Any notes about your mood? (Optional)")
        intensity = st.slider("Mood Intensity", 1, 10, 5)
        
        submitted = st.form_submit_button("Log Mood")
        
        if submitted:
            # Create mood entry
            mood_entry = {
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "emotion": selected_emotion,
                "intensity": intensity,
                "notes": mood_notes
            }
            
            # Add to log
            st.session_state.mood_log.append(mood_entry)
            st.success("Mood logged successfully!")
    
    # Display mood history
    if st.session_state.mood_log:
        st.subheader("Your Mood History")
        
        # Convert to DataFrame for easier display
        mood_df = pd.DataFrame(st.session_state.mood_log)
        st.dataframe(mood_df)
        
        # Simple visualization
        if len(mood_df) > 1:
            st.subheader("Mood Trends")
            
            # Create a numeric representation of emotions for plotting
            emotion_mapping = {emotion: i for i, emotion in enumerate(emotion_options)}
            mood_df['emotion_numeric'] = mood_df['emotion'].map(emotion_mapping)
            
            # Plot mood over time
            st.line_chart(mood_df.set_index('date')['emotion_numeric'])
    
    else:
        st.info("No mood entries yet. Start by logging your first mood!")

# Wellness Resources
elif app_mode == "Wellness Resources":
    st.header("Wellness Resources")
    
    # Get current emotion for personalized resources
    current_emotion = st.session_state.current_emotion
    
    st.markdown(f"""
    ### Personalized Wellness Resources
    Based on your current emotion: **{current_emotion}**
    """)
    
    # Provide emotion-specific resources
    provide_wellness_resources(current_emotion)

# Support functions
def provide_emotion_support(emotion):
    """Provide emotion-specific support based on facial emotion detection"""
    
    support_resources = {
        "happy": {
            "message": "It's wonderful to see you happy! Let's keep this positive energy going.",
            "affirmation": "I radiate joy and attract positivity into my life.",
            "activity": "Consider sharing your happiness with others or journaling about what's making you happy.",
            "breathing": "Try square breathing: Inhale for 4 counts, hold for 4, exhale for 4, hold for 4. Repeat 5 times."
        },
        "sad": {
            "message": "I notice you're feeling sad. It's okay to feel this way. Let's work through it together.",
            "affirmation": "This feeling is temporary, and I have the strength to move through it.",
            "activity": "Try writing about your feelings or talking to someone you trust.",
            "breathing": "Practice calming breath: Inhale slowly for 4 counts, exhale slowly for 6 counts. Repeat 10 times."
        },
        "angry": {
            "message": "I can see you're feeling angry. Let's help you find some calm.",
            "affirmation": "I acknowledge my anger without letting it control me. I choose peace.",
            "activity": "Try physical activity like walking or stretching to release tension.",
            "breathing": "Cooling breath: Inhale through your nose, exhale through your mouth as if cooling a hot drink. Repeat 8 times."
        },
        "fear": {
            "message": "I sense some fear or anxiety. Remember, you're safe right now.",
            "affirmation": "I am safe in this moment. I have the courage to face what comes.",
            "activity": "Ground yourself by naming 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste.",
            "breathing": "4-7-8 breathing: Inhale for 4 counts, hold for 7, exhale for 8. Repeat 4 times."
        },
        "surprise": {
            "message": "You seem surprised! Whether it's a good or challenging surprise, I'm here to support you.",
            "affirmation": "I am adaptable and can handle whatever comes my way.",
            "activity": "Take a moment to process what's happening before reacting.",
            "breathing": "Equal breathing: Inhale for 4 counts, exhale for 4 counts. Repeat 10 times."
        },
        "neutral": {
            "message": "You appear calm and balanced. This is a great state for mindfulness.",
            "affirmation": "I am centered and present in this moment.",
            "activity": "Practice mindfulness by focusing on your breath or surroundings.",
            "breathing": "Mindful breathing: Simply observe your natural breath without changing it for 2 minutes."
        }
    }
    
    # Default response if emotion not in our dictionary
    emotion_key = emotion.lower() if emotion.lower() in support_resources else "neutral"
    resources = support_resources[emotion_key]
    
    # Display support resources
    st.info(resources["message"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Affirmation")
        st.success(resources["affirmation"])
        
        st.subheader("Breathing Exercise")
        st.info(resources["breathing"])
    
    with col2:
        st.subheader("Suggested Activity")
        st.info(resources["activity"])
        
        # Play calming music based on emotion
        st.subheader("Calming Sounds")
        emotion_music = {
            "happy": "Upbeat instrumental",
            "sad": "Comforting classical",
            "angry": "Soothing nature sounds", 
            "fear": "Gentle piano music",
            "surprise": "Balanced ambient music",
            "neutral": "Mindfulness meditation"
        }
        st.write(f"Recommended: {emotion_music[emotion_key]}")

def provide_text_support(emotion, polarity, text):
    """Provide support based on text emotion analysis"""
    
    # Basic response based on emotion category
    if "Positive" in emotion:
        st.success("It's great to see your positive outlook! Keep nurturing those good feelings.")
        st.info("""
        **Suggestions:**
        - Share your positivity with others
        - Practice gratitude by listing 3 things you're thankful for
        - Engage in activities that bring you joy
        """)
        
    elif "Negative" in emotion:
        st.warning("I notice some challenging emotions in your words. Remember that all feelings are valid and temporary.")
        st.info("""
        **Suggestions:**
        - Practice self-compassion - be as kind to yourself as you would to a friend
        - Try journaling to process your feelings
        - Consider talking to someone you trust
        - Engage in gentle movement or time in nature
        """)
        
    else:  # Neutral
        st.info("You seem to be in a balanced state. This is a good opportunity for mindfulness and reflection.")
        st.info("""
        **Suggestions:**
        - Practice mindful breathing for 5 minutes
        - Check in with your body - any tension or discomfort?
        - Set an intention for the rest of your day
        """)
    
    # Provide a relevant quote
    quotes = {
        "positive": [
            "The only way to do great work is to love what you do. - Steve Jobs",
            "Happiness is not something ready made. It comes from your own actions. - Dalai Lama",
            "The purpose of our lives is to be happy. - Dalai Lama"
        ],
        "negative": [
            "The pain you feel today is the strength you feel tomorrow. - Unknown",
            "This too shall pass. - Persian adage", 
            "You may have to fight a battle more than once to win it. - Margaret Thatcher"
        ],
        "neutral": [
            "Peace comes from within. Do not seek it without. - Buddha",
            "The present moment is filled with joy and happiness. If you are attentive, you will see it. - Thich Nhat Hanh",
            "Feelings come and go like clouds in a windy sky. Conscious breathing is my anchor. - Thich Nhat Hanh"
        ]
    }
    
    # Select appropriate quotes
    if polarity > 0.1:
        quote_category = "positive"
    elif polarity < -0.1:
        quote_category = "negative" 
    else:
        quote_category = "neutral"
    
    st.subheader("Inspirational Quote")
    import random
    st.success(random.choice(quotes[quote_category]))

def provide_wellness_resources(emotion):
    """Provide wellness resources based on current emotion"""
    
    resources = {
        "Happy": {
            "title": "Maintaining Your Positive Energy",
            "suggestions": [
                "Practice gratitude journaling",
                "Share your happiness with others through acts of kindness",
                "Engage in activities that bring you joy",
                "Connect with supportive people in your life"
            ],
            "resources": [
                "[The Science of Happiness - Berkeley Greater Good Center](https://greatergood.berkeley.edu/)",
                "[Positive Psychology Exercises](https://positivepsychology.com/positive-psychology-exercises/)"
            ]
        },
        "Sad": {
            "title": "Navigating Sad Feelings",
            "suggestions": [
                "Allow yourself to feel without judgment",
                "Reach out to trusted friends or family",
                "Engage in gentle self-care activities",
                "Consider professional support if sadness persists"
            ],
            "resources": [
                "[Crisis Text Line](https://www.crisistextline.org/)",
                "[Mental Health America Resources](https://www.mhanational.org/)"
            ]
        },
        "Angry": {
            "title": "Managing Anger Constructively", 
            "suggestions": [
                "Take a timeout before responding",
                "Practice deep breathing or physical activity",
                "Identify the underlying need or boundary",
                "Express feelings using 'I' statements"
            ],
            "resources": [
                "[Anger Management Techniques](https://www.apa.org/topics/anger/control)",
                "[Mindful Approaches to Anger](https://www.mindful.org/a-mindful-approach-to-anger/)"
            ]
        },
        "Anxious": {
            "title": "Calming Anxiety",
            "suggestions": [
                "Practice grounding techniques (5-4-3-2-1 method)",
                "Limit caffeine and get adequate sleep",
                "Break tasks into manageable steps",
                "Challenge catastrophic thinking patterns"
            ],
            "resources": [
                "[Anxiety and Depression Association of America](https://adaa.org/)",
                "[Calm Breathing Exercises](https://www.calm.com/breathe)"
            ]
        },
        "Neutral": {
            "title": "Cultivating Mindfulness",
            "suggestions": [
                "Practice present-moment awareness",
                "Try a mindfulness meditation",
                "Engage in activities with full attention",
                "Notice sensations in your body without judgment"
            ],
            "resources": [
                "[Mindfulness Exercises](https://www.mindful.org/meditation/mindfulness-getting-started/)",
                "[Headspace Guided Meditations](https://www.headspace.com/)"
            ]
        }
    }
    
    # Default to neutral if emotion not found
    emotion_key = emotion if emotion in resources else "Neutral"
    resource = resources[emotion_key]
    
    st.subheader(resource["title"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Suggestions**")
        for suggestion in resource["suggestions"]:
            st.write(f"• {suggestion}")
    
    with col2:
        st.markdown("**Additional Resources**")
        for link in resource["resources"]:
            st.markdown(link)
    
    # Breathing exercise for all emotions
    st.subheader("Quick Breathing Exercise")
    st.markdown("""
    **Box Breathing Technique:**
    1. Inhale slowly for 4 counts
    2. Hold your breath for 4 counts  
    3. Exhale slowly for 4 counts
    4. Hold empty for 4 counts
    5. Repeat 5-10 times
    """)
    
    # Progress bar for breathing exercise
    if st.button("Start 2-Minute Breathing Exercise"):
        breathing_placeholder = st.empty()
        for i in range(4):  # 4 cycles of box breathing
            for phase, text in [("Inhale", "🟢 INHALE"), ("Hold", "🟡 HOLD"), ("Exhale", "🔴 EXHALE"), ("Hold", "🟡 HOLD")]:
                breathing_placeholder.info(f"{text} for 4 seconds")
                time.sleep(4)
        breathing_placeholder.success("Breathing exercise complete! Notice how you feel.")

# Footer
st.markdown("---")
st.markdown("""
**AI MoodCare** - Your mental wellness companion. 
Remember: This app is not a substitute for professional mental health care. 
If you're experiencing a crisis, please contact a mental health professional or crisis helpline.
""")

if __name__ == "__main__":
    # This is already a Streamlit app, so no need to run anything else
    pass