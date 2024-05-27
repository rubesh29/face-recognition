import streamlit as st
from Home import face_rec
st.set_page_config(page_title='Reporting',layout='wide')
st.subheader('Attendance Report')

#extract data from redis lisi
name='attendance:logs'
def load_logs(name,end=-1):
    logs_list=face_rec.r.lrange(name,start=0,end=end)
    return logs_list
tab1,tab2=st.tabs(['Registered data','logs'])
with tab1:
    if st.button('refresh data'):
        with st.spinner('Retriving Data from redis db'):
            redis_face_db = face_rec.retrive_data(name='academy:register')
            st.dataframe(redis_face_db[['name','role']])

with tab2:            
    if st.button('Refresh'):
        st.write(load_logs(name=name))
        
