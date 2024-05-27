import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer
import av
from Home import face_rec

st.set_page_config(page_title='Registration',layout='centered')
st.subheader('Register')

#initilaize registration form
registraction_form=face_rec.RegistrationForm()

#collect person name and role
person_name=st.text_input(label='Name',placeholder='first and last Name')
person_role=st.selectbox(label='Role',options=('Student','Teacher'))

#collect facial emmbedings
from streamlit_webrtc import webrtc_streamer
import av


def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")#3d array
    reg_img,embedding = registraction_form.get_embedding(img)

    #save embedding in local file  txt
    if embedding is not None:
        with open('face_embedding.txt',mode='ab') as f:
            np.savetxt(f,embedding)

    return av.VideoFrame.from_ndarray(reg_img, format="bgr24")


webrtc_streamer(key="registration", video_frame_callback=video_frame_callback)



if st.button('submit'):
    return_val =registraction_form.save_data_in_redis(person_name,person_role)
    if return_val==True:
        st.success(f'{person_name} registered sussfully')

    elif return_val=='name is false':
        st.error('please enter the name')
    elif return_val=='file fales':
        st.error('face_embedding.txt not found')        
