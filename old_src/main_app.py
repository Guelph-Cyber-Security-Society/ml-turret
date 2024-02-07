import streamlit as st
import cv2
from streamlit_mic_recorder import mic_recorder,speech_to_text #scuffed ass random library that works surprisingly well
from object_detector import *


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
    st.title("Webcam Live Feed")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    camera_feed = process_video(1) #param 1 is for local webcam, unsure what path for pi cam

    while run:
        frame = next(camera_feed)
        FRAME_WINDOW.image(frame, channels='BGR')
    else:
        st.write('Stopped')

if __name__ == "__main__":
    main()
