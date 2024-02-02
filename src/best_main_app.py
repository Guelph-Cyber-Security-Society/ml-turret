import streamlit as st
import cv2
from streamlit_mic_recorder import mic_recorder,speech_to_text #scuffed ass random library that works surprisingly well
import sys
import os

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the models directory
models_dir = os.path.join(current_dir, 'models')

# Add the models directory to the sys.path
sys.path.append(models_dir)

# Now you can import from the models directory
from myMentalHealthIsDown import process_frame

def main():
    state=st.session_state
    if 'text_received' not in state:
        state.text_received=[]

    st.title("Streamlit Interface")

    # Text input and submission
    text_input = st.text_input("Enter your text here:")
    submit_button = st.button("Submit Text")
    if submit_button:
        st.write(f"You submitted: {text_input}") #This would be passed to a LLM

    # Microphone input (workaround)
    st.write("Upload a voice recording (as a workaround for microphone input):")
    text=speech_to_text(language='en',use_container_width=True,just_once=True,key='STT')
    if text:       
        state.text_received.append(text)

    for text in state.text_received:
        st.text(text) #Find a way to process this via LLM

    paces = st.number_input("Enter number of paces:", min_value=1, step=1, format="%i")

    #Manual controls - nonfunctional obv
    st.write("Directional Buttons:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Left"):
            #rover.manualRoverMove(paces, "Left")
            pass
    with col2:
        if st.button("Up"):
            #rover.manualRoverMove(paces, "Up")
            pass
        if st.button("Down"):
            #rover.manualRoverMove(paces, "Down")
            pass
    with col3:
        if st.button("Right"):
            #rover.manualRoverMove(paces, "Right")
            pass
    
    #hell on earth - do not fuck with
    #universel webcam live feed code regardless of backend
    st.title("Object Detection with Streamlit")

    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(1)  # Use 0 for webcam
        while run:
            ret, frame = cap.read()
            if not ret:
                st.write("Error: Unable to fetch frame.")
                break
            processed_frame = process_frame(frame)
            FRAME_WINDOW.image(processed_frame, channels='BGR', use_column_width=True)
            if st.button('Stop'):
                break
        cap.release()
    else:
        st.write('Stopped')


if __name__ == "__main__":
    main()
