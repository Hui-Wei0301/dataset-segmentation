# on another CPU machine
from bert_serving.client import BertClient
import os

bc = BertClient()  # ip address of the GPU machine
train_path = r'/home/weihui/huiwei/baseline/text-segmentation-master/data/WikiSection/wikisection_dataset_ref/en_city_train'
fileList = os.listdir(train_path)
for file in fileList:
    path = os.path.join(train_path, file)
    f = open(path,'r')
    sentences = f.readlines()
    for sentence in sentences:
        embedding = bc.encode(sentence)
        with open(r'~\huiwei\bert\bert_emb_train','a+') as f1:
            f1.write(embedding+'\n')
    f1.write('\n')
    f.close()
f1.close()
# bc.encode(['First do it', 'then do it right', 'then do it better'])
