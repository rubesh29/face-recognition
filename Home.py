import streamlit as st
import streamlit_authenticator as stauth
import face_rec

st.set_page_config(page_title='Attendance_system',layout='wide')




st.header('Attendance System Using Face Recognition')

with st.spinner("Loading models and connecting to redis db..."):
    import face_rec
st.success('model loaded sucessfully')
st.success('Redis db is sucessfuly connected')    


