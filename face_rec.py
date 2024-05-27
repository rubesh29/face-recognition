import numpy as np
import pandas as pd
import cv2
import redis
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise
import time
from datetime import datetime
import os

#connecting to redis cloud
r = redis.StrictRedis(host='redis-13543.c14.us-east-1-2.ec2.cloud.redislabs.com',port=13543,password='s8CUbhz53j45twnIGtLYyhQvfv6h3wQM')

#retrive data from database
def retrive_data(name):
    retrive_dict=r.hgetall(name)
    retrive_series = pd.Series(retrive_dict)
    retrive_series =retrive_series.apply(lambda x: np.frombuffer(x,dtype=np.float32))
    index=retrive_series.index
    index=list(map(lambda x:x.decode(),index))
    retrive_series.index=index
    retrive_df=retrive_series.to_frame().reset_index()
    retrive_df.columns=['name_role','facial_features']
    retrive_df[['name','role']]=retrive_df['name_role'].apply(lambda x:x.split('@')).apply(pd.Series)
    return retrive_df[['name','role','facial_features']]

#cofigure our face analysis

faceapp= FaceAnalysis(name='buffalo_l',
                      root='insightface1_model',
                      providers=['CPUExecutionProvider'])

faceapp.prepare(ctx_id=0 ,det_size=(640,640),det_thresh=0.5)

#ml search algo
def ml_search_algorithm(dataframe,feature_column,test_vector,name_role=['NAME','ROLE'],thresh=0.5):
    #step1:take a data frame
    dataframe=dataframe.copy()
    #step2:take face embedding from datafram and covert it into an np array
    x_list=dataframe[feature_column].tolist()
    x=np.asarray(x_list)
    #step3:calulatin the cosine similarity
    similar=pairwise.cosine_similarity(x,test_vector.reshape(1,-1))
    similar_arr=np.array(similar).flatten()
    dataframe['cosine']=similar_arr
    #step4:filter the data
    data_filter=dataframe.query(f'cosine >= {thresh}')
    if len(data_filter)>0:
        data_filter.reset_index(drop=True,inplace=True)
        argmax=data_filter['cosine'].argmax()
        person_name,person_role=data_filter.loc[argmax][name_role]

    else:
        person_name='unknown'
        person_role='unknown'

    return person_name , person_role

#we need to save logs for 1 mins
class RealTimePrediction:
    def __init__(self):
        self.logs=dict(name=[],role=[],current_time=[])
    def reset(self):
        self.logs=dict(name=[],role=[],current_time=[])
    def saveLogs_redis(self):
        #creat a lod dataframe
        dataframe=pd.DataFrame(self.logs)
        #drop the dupilicates
        dataframe.drop_duplicates('name',inplace=True)
        #push it in redis database in list(it takes only one value)
        name_list=dataframe['name'].to_list()
        role_list=dataframe['role'].to_list()
        ctime_list=dataframe['current_time'].to_list()
        encoded_data=[]

        for name,role,ctime in zip(name_list,role_list,ctime_list):
            if name !='unknown':
                concat_string=f"{name}@{role}@{ctime}"
                encoded_data.append(concat_string)

        if len(encoded_data)>0:
            r.lpush('attendance:logs',*encoded_data)

        self.reset()    


    #dedect the multiple person
    def face_prediction(self,test_image,dataframe,feature_column,name_role=['NAME','ROLE'],thresh=0.5):
        current_time=str(datetime.now())
        results=faceapp.get(test_image)
        test_copy=test_image.copy()
        #use ml search algo
        for res in results:
            x1,y1,x2,y2=res['bbox'].astype(int)
            embeddings=res['embedding']
            person_name,person_role= ml_search_algorithm(dataframe,feature_column,test_vector=embeddings,name_role=name_role,thresh=thresh)
            print(person_name,person_role)

            if person_name=='unknown':
                color=(0,0,225)
            else:
                color=(0,225,0)

            cv2.rectangle(test_copy,(x1,y1),(x2,y2),color)
            text_gen = person_name
            cv2.putText(test_copy,text_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
            cv2.putText(test_copy,current_time,(x1,y2+10),cv2.FONT_HERSHEY_DUPLEX,0.7,color,2)
            #save info in logs dict
            self.logs['name'].append(person_name)
            self.logs['role'].append(person_role)
            self.logs['current_time'].append(current_time)


        return test_copy       

class RegistrationForm:
    def __init__(self):
        self.sample=0
    def reset_sam(self):
        self.sample=0
    def get_embedding(self,frame):
        results=faceapp.get(frame,max_num=1)
        embeddings=None
        for res in results:
            self.sample +=1
            x1,y1,x2,y2=res['bbox'].astype(int)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
            #sample info
            text=f'samples={self.sample}'
            cv2.putText(frame,text,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,0,2))

            embeddings=res['embedding']

        return frame,embeddings    
    
    def save_data_in_redis(self,name,role):
        if name is not None:
            if name.strip() !='':
                key =f'{name}@{role}'
            else:
                return 'name is false' 

        else:
            return 'name is false'

        if 'face_embedding.txt' not in os.listdir():
            return 'file fales'

        #load embeding txt
        x_array=np.loadtxt('face_embedding.txt',dtype=np.float32) 
        recived_samples=int(x_array.size/512)
        x_array=x_array.reshape(recived_samples,512)
        x_array=np.asarray(x_array)
        x_mean=x_array.mean(axis=0)
        x_mean=x_mean.astype(np.float32)
        x_mean_bytes=x_mean.tobytes()

        r.hset(name='academy:register',key=key,value=x_mean_bytes)

        os.remove('face_embedding.txt')
        self.reset_sam()
        return True
            