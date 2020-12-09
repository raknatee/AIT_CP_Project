import pickle
import os
import json

des_path = r"/home/jovyan/work/root/datasets/001_json_version"
ori_path = r"/home/jovyan/work/root/ori_datasets"
filenames = os.listdir(ori_path)
print(filenames)

def create_filename(i,max_digi=2):
    d = len(str(i))
    if(d<max_digi):
        return '0'*(max_digi-d)+str(i)
    else:
        return str(i)
    


for i,filename in enumerate(filenames):
    print(f"doing iter: {i} {filename}")
    with open(os.path.join(des_path,f"person_{create_filename(i)}.json"),"w") as wf:
        with open(f"{ori_path}/{filename}","rb") as f:
            
            temp = {'data':None,'labels':None}
            
            data = pickle.load(f,encoding='bytes')
            
            temp['data'] = data[b'data'].tolist()
            # print(temp['data'].tolist())
            temp['labels'] = data[b'labels'].tolist()
            # print(temp)
            json.dump(temp,wf)