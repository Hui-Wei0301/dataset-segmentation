# on another CPU machine
from bert_serving.client import BertClient
import os

bc = BertClient(check_length=False)  # ip address of the GPU machine
train_path = r'/home/weihui/huiwei/baseline/text-segmentation-master/data/transcript/en_disease_train_transcript'
fileList = os.listdir(train_path)
for file in fileList:
    path = os.path.join(train_path, file)
    f = open(path,'r')
    sentences = f.readlines()
    embeddings = bc.encode(sentences)
    for embedding in embeddings:
        f1 = open(r'/home/weihui/huiwei/bert/bert_emb_dev', 'a+')
        f1.writelines(str(embedding))
    f1.write('\n')
    f.close()
f1.close()
# bc.encode(['First do it', 'then do it right', 'then do it better'])
