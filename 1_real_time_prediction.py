import streamlit as st
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time
import cv2


st.set_page_config(page_title='real_time_face_presiction')


st.subheader('Face Prediction')


#retrive the data from  redis database
with st.spinner('Retriving Data from redis db'):
    redis_face_db = face_rec.retrive_data(name='academy:register')
    st.dataframe(redis_face_db)

st.success('data sucessfully retrived from redis')  


#time
waittime=30
setTime=time.time()
realtimepred=face_rec.RealTimePrediction()#class
#real time prediction
#using web-rtc


def video_frame_callback(frame):
    global setTime
    img = frame.to_ndarray(format="bgr24")

    pred_frame=realtimepred.face_prediction(img,redis_face_db,'facial_features',['name','role'],thresh=0.5)
    timenow=time.time()
    difftime=timenow-setTime
    if difftime>=waittime:
        realtimepred.saveLogs_redis()
        setTime=time.time()
    return av.VideoFrame.from_ndarray(pred_frame, format="bgr24")
    
    


webrtc_streamer(key="Faceprediction", video_frame_callback=video_frame_callback)



