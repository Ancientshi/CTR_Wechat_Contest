# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pandas.core.reshape.concat import concat
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models.deepfm import *
from deepctr_torch.models.basemodel import *
from of_demo.contest.tool.uAuc2 import uAUC as uAUC2
from of_demo.contest.tool.map import mapId
# 存储数据的根目录
ROOT_PATH = "/home/shiyunxiao/deepCTR/wechat_big_data_baseline_pytorch/data"
# 比赛数据集路径
DATASET_PATH = ROOT_PATH + '/wechat_algo_data1/'
# 训练集
USER_ACTION = DATASET_PATH + "user_action.csv"
FEED_INFO = DATASET_PATH + "feed_info.csv"
FEED_EMBEDDINGS = DATASET_PATH + "feed_embeddings.csv"
# 测试集
TEST_FILE = DATASET_PATH + "test_a.csv"
# 初赛待预测行为列表
ACTION_LIST = ["read_comment", "like", "click_avatar", "forward"]
FEA_COLUMN_LIST = ["read_comment", "like", "click_avatar", "forward", "comment", "follow", "favorite"]
FEA_FEED_LIST = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']

# 负样本下采样比例(负样本:正样本)
ACTION_SAMPLE_RATE = {"read_comment": 5, "like": 5, "click_avatar": 5, "forward": 10, "comment": 10, "follow": 10,
                      "favorite": 10}


device_ids = [0, 1, 2, 3]

class MyBaseModel(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, l2_reg_linear, l2_reg_embedding, init_std, seed, task, device, gpus,rate=0):
        #gpus=device_ids
        super().__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear, l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)
        self.rate=rate
        self.keyword_embedding_table = nn.Embedding(200000, 8)
        self.tag_embedding_table = nn.Embedding(2000, 8)

    def fit(self, x=None,y=None, batch_size=None, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,validation_data=None, shuffle=True, callbacks=None,other=None):

        train_textDict=other['train_textDict']
        train_manual_keyword_list=np.array(train_textDict['manual_keyword_list']).reshape(-1,6)
        train_machine_keyword_list=np.array(train_textDict['machine_keyword_list']).reshape(-1,6)
        train_itsc_keyword_list=np.array(train_textDict['itsc_keyword_list']).reshape(-1,3)
        train_manual_tag_list=np.array(train_textDict['manual_tag_list']).reshape(-1,6)
        train_machine_tag_list=np.array(train_textDict['machine_tag_list']).reshape(-1,6)
        train_itsc_tag_list=np.array(train_textDict['itsc_tag_list']).reshape(-1,3)



        # userid
        # feedid
        # authorid
        # bgm_song_id
        # bgm_singer_id
        # videoplayseconds
        # feedembedding_0
        # feedembedding_1
        # for feature in self.feature_index:
        #     print(feature)
    
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]

        do_validation = False
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
            else:
                raise ValueError(
                    'When passing a `validation_data` argument, '
                    'it must contain either 2 items (x_val, y_val), '
                    'or 3 items (x_val, y_val, val_sample_weights), '
                    'or alternatively it could be a dataset or a '
                    'dataset or a dataset iterator. '
                    'However we received `validation_data=%s`' % validation_data)
            if isinstance(val_x, dict):
                val_x = [val_x[feature] for feature in self.feature_index]

        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            if hasattr(x[0], 'shape'):
                split_at = int(x[0].shape[0] * (1. - validation_split))
            else:
                split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_arrays(x, 0, split_at),
                        slice_arrays(x, split_at))
            y, val_y = (slice_arrays(y, 0, split_at),
                        slice_arrays(y, split_at))
        else:
            val_x = []
            val_y = []
        
        #len(x)为70
        for i in range(len(x)):
            #print(x[i].shape) (4724,)
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
                #print(x[i].shape) (4724, 1)
        x=np.concatenate(x, axis=-1)
        #print('x.shape',x.shape) (4724, 70)

        x=np.concatenate([x,train_manual_keyword_list,train_machine_keyword_list,train_itsc_keyword_list,
        train_manual_tag_list,train_machine_tag_list,train_itsc_tag_list],axis=1)
        train_tensor_data = Data.TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(y))
        # len(train_tensor_data[0]) 2
        # train_tensor_data[0][0].shape torch.Size([70])
        # train_tensor_data[0][0][0].shape torch.Size([])
        # print('len(train_tensor_data[0])',len(train_tensor_data[0]))
        # print('train_tensor_data[0][0].shape',train_tensor_data[0][0].shape) 
        # print('train_tensor_data[0][0][0].shape',train_tensor_data[0][0][0].shape) 
        #print('y.shape',train_tensor_data[1].shape)
        
        if batch_size is None:
            batch_size = 256

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim

        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            print(self.device)

        train_loader = DataLoader(
            dataset=train_tensor_data, shuffle=shuffle, batch_size=batch_size)

        sample_num = len(train_tensor_data)
        #根据batch_size，一个epoch要多少次
        steps_per_epoch = (sample_num - 1) // batch_size + 1

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        # Train
        print("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_tensor_data), len(val_y), steps_per_epoch))
        #训练epochs次
        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            loss1_epoch = 0
            loss2_epoch = 0
            loss3_epoch = 0
            loss4_epoch = 0
            total_loss_epoch = 0
            
            train_result = {}
            #开始训练一个批次
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for _, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()
                        # print('x.shape',x.shape) [512, 6]
                        # print('y.shape',y.shape) [512, 4]
                        y_pred = model(x).squeeze()
                        #print('y_pred.shape',y_pred.shape) [512]
                        optim.zero_grad()
                        #分割长度，分割维度
                        label=y.squeeze()
                        y_pred1,y_pred2,y_pred3,y_pred4=y_pred.split(1,dim=1)
                        label1,label2,label3,label4=label.split(1,dim=1)
                        loss1 = loss_func(y_pred1, label1, reduction='sum')
                        loss2 = loss_func(y_pred2, label2, reduction='sum')
                        loss3 = loss_func(y_pred3, label3, reduction='sum')
                        loss4 = loss_func(y_pred4, label4, reduction='sum')
                        loss = loss_func(y_pred, y.squeeze(), reduction='sum')
                        
                        reg_loss = self.get_regularization_loss()

                        total_loss = loss + reg_loss + self.aux_loss

                        loss_epoch += loss.item()
                        total_loss_epoch += total_loss.item()

                        loss1_epoch+=loss1.item()
                        loss2_epoch+=loss2.item()
                        loss3_epoch+=loss3.item()
                        loss4_epoch+=loss4.item()

                        total_loss.backward()

                        


                        optim.step()

                        if verbose > 0:
                            for name, metric_fun in self.metrics.items():
                                if name not in train_result:
                                    train_result[name] = []
                                try:
                                    temp = metric_fun(
                                        y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64"))
                                except Exception:
                                    temp = 0
                                finally:
                                    train_result[name].append(temp)
            except KeyboardInterrupt:
                t.close()
                raise
            t.close()
            #一个批次训练完了

            # Add epoch_logs
            epoch_logs["total_loss"] = total_loss_epoch / sample_num
            epoch_logs["weight_loss"] = (0.4*loss1_epoch+0.3*loss2_epoch+0.2*loss3_epoch+0.1*loss4_epoch) / (4*sample_num)
            epoch_logs["loss1"] = loss1_epoch / sample_num
            epoch_logs["loss2"] = loss2_epoch / sample_num
            epoch_logs["loss3"] = loss3_epoch / sample_num
            epoch_logs["loss4"] = loss4_epoch / sample_num
            
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / steps_per_epoch

            if do_validation:
                eval_result = self.evaluate(val_x,other, val_y, batch_size)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
                
                #计算uAuc
                val_x, val_y = validation_data
                size=val_y.shape[0]
                #(batch_size, 1)
                #print('val_y.shape',val_y.shape)
                #(batch_size, )
                #print('val_x.shape',val_x['userid'].values.shape)
                predict=self.predict(val_x, other,size).reshape(-1,1)
                predict=predict.reshape(-1,4)
                label=val_y.reshape(-1,4)
                weight_uAuc,uAUC =uAUC2(label,predict,val_x['userid'].values.flatten())
                
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                print('Epoch {0}/{1}'.format(epoch + 1, epochs))
                print("weight_uAuc:",weight_uAuc,"uAuc1:",uAUC[0],"uAuc2:",uAUC[1],"uAuc3:",uAUC[2],"uAuc4:",uAUC[3])
                record[str(self.rate)][str(epoch)]['uAuc']="%s,%s,%s,%s,%s"%(weight_uAuc,uAUC[0],uAUC[1],uAUC[2],uAUC[3])
                eval_str=''
                for lossname,value in epoch_logs.items():
                    eval_str += "{0}s - {1}: {2: .4f}".format(
                        epoch_time,lossname, value)

                for name in self.metrics:
                    eval_str += " - " + name + \
                                ": {0: .4f}".format(epoch_logs[name])

                if do_validation:
                    for name in self.metrics:
                        eval_str += " - " + "val_" + name + \
                                    ": {0: .4f}".format(epoch_logs["val_" + name])
                print(eval_str)
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()

        return self.history

    def evaluate(self, x, other,y, batch_size=256):
        """

        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        pred_ans = self.predict(x,other, batch_size)
        eval_result = {}
        for name, metric_fun in self.metrics.items():
            try:
                temp = metric_fun(y, pred_ans)
            except Exception:
                temp = 0
            finally:
                eval_result[name] = metric_fun(y, pred_ans)
        return eval_result



    def predict(self, x,other, batch_size=256):
        """

        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        valid_textDict=other['valid_textDict']
        valid_manual_keyword_list=np.array(valid_textDict['manual_keyword_list']).reshape(-1,6)
        valid_machine_keyword_list=np.array(valid_textDict['machine_keyword_list']).reshape(-1,6)
        valid_itsc_keyword_list=np.array(valid_textDict['itsc_keyword_list']).reshape(-1,3)
        valid_manual_tag_list=np.array(valid_textDict['manual_tag_list']).reshape(-1,6)
        valid_machine_tag_list=np.array(valid_textDict['machine_tag_list']).reshape(-1,6)
        valid_itsc_tag_list=np.array(valid_textDict['itsc_tag_list']).reshape(-1,3)



        model = self.eval()
        if isinstance(x, dict):
            x = [x[feature] for feature in self.feature_index]
        for i in range(len(x)):
            if len(x[i].shape) == 1:
                x[i] = np.expand_dims(x[i], axis=1)
        x=np.concatenate(x, axis=-1)
        x=np.concatenate([x,valid_manual_keyword_list,valid_machine_keyword_list,valid_itsc_keyword_list,
        valid_manual_tag_list,valid_machine_tag_list,valid_itsc_tag_list],axis=1)
        data=torch.from_numpy(x)
        tensor_data = Data.TensorDataset(data)
        test_loader = DataLoader(
            dataset=tensor_data, shuffle=False, batch_size=batch_size)

        pred_ans = []
        with torch.no_grad():
            for _, x_test in enumerate(test_loader):
                x = x_test[0].to(self.device).float()

                y_pred = model(x).cpu().data.numpy()  # .squeeze()
                pred_ans.append(y_pred)

        return np.concatenate(pred_ans).astype("float64")

class MyDeepFM(MyBaseModel):
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, use_fm=True,
                 dnn_hidden_units=(256,128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='gpu', gpus=None,rate=0):


        super(MyDeepFM, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                     l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                     device=device, gpus=gpus,rate=rate)

        self.use_fm = use_fm
        self.use_dnn = 1
        if use_fm:
            self.fm = FM()

        if self.use_dnn:
            # inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,init_std=0.0001, dice_dim=3, seed=1024, device='cpu'
            
            #DNN层
            inputDim=self.compute_input_dim(dnn_feature_columns)+(6+6+3)*8+(6+6+3)*8
            self.dnn = DNN(inputDim, hidden_units=dnn_hidden_units,
                           activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                           init_std=init_std, device=device)
            #全连接层
            self.fn1 = nn.Linear(
                #in out bias
                dnn_hidden_units[-1], 8, bias=False)

            #全连接层
            self.fn2 = nn.Linear(
                #in out bias
                dnn_hidden_units[-1], 8, bias=False)
            #全连接层
            self.fn3 = nn.Linear(
                #in out bias
                dnn_hidden_units[-1], 8, bias=False)

            #全连接层
            self.fn4 = nn.Linear(
                #in out bias
                dnn_hidden_units[-1], 8, bias=False)

            #mmoe层
            self.mmoe1 = nn.Linear(
                #in out bias
                8, 1, bias=False)

            #mmoe层
            self.mmoe2 = nn.Linear(
                #in out bias
                8, 1, bias=False)

            #mmoe层
            self.mmoe3 = nn.Linear(
                #in out bias
                8, 1, bias=False)

            #mmoe层
            self.mmoe4 = nn.Linear(
                #in out bias
                8, 1, bias=False)

            # #对DNN的正则
            # self.add_regularization_weight(
            #     #filter() 函数用于过滤序列，过滤掉不符合条件的元素，返回由符合条件元素组成的新列表。
            #     weight_list=filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
            # #对全连接的正则
            # self.add_regularization_weight(self.dnn_linear.weight, l2=l2_reg_dnn)
        self.to(device)



    def forward(self, X):
        #[512, 70+(6+6+3)+(6+6+3)]
        #print('x.shape',X.shape)

        #sparse里面每一个[512, 1, 128]，有5个，dense里面每一个[512, 1]，有65个，构成128*5+65=705
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X[:,0:70], self.dnn_feature_columns,
                                                                                  self.embedding_dict)
        #train_manual_keyword_list,train_machine_keyword_list,train_itsc_keyword_list,train_manual_tag_list,train_machine_tag_list,train_itsc_tag_list

        #X是tensor类型
        keyword_array=torch.LongTensor(X[:,70:85].cpu().numpy()).cuda(self.device)
        tag_array=torch.LongTensor(X[:,85:].cpu().numpy()).cuda(self.device)
        keyword_embedding=self.keyword_embedding_table(keyword_array)
        tag_embedding=self.tag_embedding_table(tag_array)
        keyword_embedding=torch.reshape(keyword_embedding,(X.shape[0],-1))
        tag_embedding=torch.reshape(tag_embedding,(X.shape[0],-1))

        #torch.Size([512, 1, 128])
        #print('sparse_embedding_list.shape',sparse_embedding_list[-1].shape) 
        #torch.Size([512, 1])
        #print('dense_value_list.shape',dense_value_list[-1].shape) 
        
        logit = self.linear_model(X)

        if self.use_fm and len(sparse_embedding_list) > 0:
            #使用fm只需要传入sparse
            fm_input = torch.cat(sparse_embedding_list, dim=1)
            logit += self.fm(fm_input)

        if self.use_dnn:
            
            #得到dnn的input
            sparse_embedding = combined_dnn_input(sparse_embedding_list, dense_value_list)
            # sparse_embedding.shape torch.Size([396, 705])
            # print('sparse_embedding.shape',sparse_embedding.shape)
            #torch.Size([512, 960])
            # print('keyword_embedding.shape',keyword_embedding.shape)
            #torch.Size([512, 480])
            # print('tag_embedding.shape',tag_embedding.shape)
            dnn_input=torch.cat([sparse_embedding,keyword_embedding,tag_embedding],dim=1)


                    
            dnn_output = self.dnn(dnn_input)

            fn1_output = self.fn1(dnn_output)
            fn2_output = self.fn2(dnn_output)
            fn3_output = self.fn3(dnn_output)
            fn4_output = self.fn4(dnn_output)

            logit1=self.mmoe1(fn1_output)
            logit2=self.mmoe1(fn2_output)
            logit3=self.mmoe1(fn3_output)
            logit4=self.mmoe1(fn4_output)
            #mmoe的logit
            mmoe_logit=torch.cat((logit1, logit2, logit3,logit4), 1)
            #线性部分的Logit
            logit=torch.cat((logit, logit, logit,logit), 1)
            #加到一起
            logit += mmoe_logit

        y_pred = self.out(logit)

        return y_pred

def listPadding(array=None,length=10):
    if len(array)>length:
        return array[:length]
    elif len(array)<length:
        return array+[0]*(length-len(array))
    else:
        return array


submit = pd.read_csv(ROOT_PATH + '/test_data.csv')[['userid', 'feedid']]
mapid=mapId()
feedEmbedding_DIM=64
mapid.pca_feedEmbedding(dim=feedEmbedding_DIM)
def handle_Keyword(manual_keyword_df,machine_keyword_df):
    manual_keyword_list=manual_keyword_df.values.flatten().tolist()
    machine_keyword_list=machine_keyword_df.values.flatten().tolist()
    manual_keywords_array=[]
    machine_keywords_array=[]
    itsc_keywords_array=[]
    for i in tqdm(range(len(machine_keyword_list))):
        str_manual_keywords=manual_keyword_list[i]
        str_machine_keywords=machine_keyword_list[i]
        if type(str_manual_keywords) is float:
            manual_keywords=[0]
        else:
            manual_keywords=list(map(lambda x: int(x)+1,str_manual_keywords.split(';')))
        if type(str_machine_keywords) is float:
            machine_keywords=[0]
        else:
            try:
                machine_keywords=list(map(lambda x: int(x)+1,str_machine_keywords.split(';')))
            except:
                print(str_machine_keywords)
                return
        itsc_keywords=list(set(manual_keywords).intersection(set(machine_keywords)))

        manual_keywords_array.append(listPadding(manual_keywords,6))
        machine_keywords_array.append(listPadding(machine_keywords,6))
        itsc_keywords_array.append(listPadding(itsc_keywords,3))
    return manual_keywords_array,machine_keywords_array,itsc_keywords_array
    #return np.array(manual_keywords_array),np.array(machine_keywords_array),np.array(itsc_keywords_array)
def handle_Tag(manual_tag_df,machine_tag_df):
    manual_tag_list=manual_tag_df.values.flatten().tolist()
    machine_tag_list=machine_tag_df.values.flatten().tolist()
    manual_tags_array=[]
    machine_tags_array=[]
    itsc_tags_array=[]
    for i in tqdm(range(len(machine_tag_list))):
        str_manual_tags=manual_tag_list[i]
        str_machine_tags=machine_tag_list[i]
        if type(str_manual_tags) is float:
            manual_tags=[0]
        else:
            manual_tags=list(map(lambda x: int(x)+1,str_manual_tags.split(';')))
        if type(str_machine_tags) is float:
            machine_tags=[0]
        else:
            tagid_rate_list=str_machine_tags.split(';')
            machine_tags=[]
            for tagid_rate in tagid_rate_list:
                tag,rate=tagid_rate.split(' ')
                if float(rate)>=0.5:
                    tag=int(tag)+1
                    machine_tags.append(tag)
        itsc_tags=list(set(manual_tags).intersection(set(machine_tags)))

        manual_tags_array.append(listPadding(manual_tags,6))
        machine_tags_array.append(listPadding(machine_tags,6))
        itsc_tags_array.append(listPadding(itsc_tags,3))
    
    return manual_tags_array,machine_tags_array,itsc_tags_array
    #return np.array(manual_tags_array),np.array(machine_tags_array),np.array(itsc_tags_array)
def getFeed_embedding(feedidDF=0):
    feedidList=feedidDF.values.tolist()
    feed_embedding=[]
    print("prepare feedEmbedding_%s"%feedEmbedding_DIM)
    for feedid in tqdm(feedidList):
        
        e=mapid.getFeedEmbedding(id=feedid)
        feed_embedding.append(e.tolist())
    feed_embedding=np.array(feed_embedding).reshape(-1,feedEmbedding_DIM)
    #print('load feed_embedding finish!feed_embedding.shape',feed_embedding.shape) (7739867,256)
    return feed_embedding


record={}

for actions in [ACTION_LIST]:
    USE_FEAT = ['userid', 'feedid']+actions + FEA_FEED_LIST[1:]

    # train = pd.read_csv(ROOT_PATH + '/allData.csv')[USE_FEAT+['date_']+['manual_keyword_list','machine_keyword_list','manual_tag_list','machine_tag_list']]
    # #frac取样100%，如果添加参数 reset_index(drop=True) 那么原index会被丢弃，不会显示为一个新列。
    # #train = train.sample(frac=0.1, random_state=42).reset_index(drop=True)
    # #除了action以外的东西
    # test = pd.read_csv(ROOT_PATH + '/test_data.csv',nrows=10)
    # for action in actions:
    #     test[action] = 0
    # data = pd.concat((train, test)).reset_index(drop=True)


    #处理feed_embedding
    # feed_embedding=getFeed_embedding(data['feedid'])
    # print('append df columns')
    # for i in tqdm(range(64)):
    #     feedEmbedding_features.append('feedembedding_%s'%i)
    #     data['feedembedding_%s'%i]=feed_embedding[:,i:i+1].flatten().tolist()

    #处理文本
    # manual_keywords_array,machine_keywords_array,itsc_keywords_array=handle_Keyword(data['manual_keyword_list'],data['machine_keyword_list'])
    # manual_tags_array,machine_tags_array,itsc_tags_array=handle_Tag(data['manual_tag_list'],data['machine_tag_list'])
    
    # data['manual_keyword_list']=manual_keywords_array
    # data['machine_keyword_list']=machine_keywords_array
    # data['itsc_keyword_list']=itsc_keywords_array

    # data['manual_tag_list']=manual_tags_array
    # data['machine_tag_list']=machine_tags_array
    # data['itsc_tag_list']=itsc_tags_array

    '''
    保存上述处理文件
    '''
    # data.iloc[:train.shape[0]].to_pickle(ROOT_PATH+'/allData2.pkl')
    # data.iloc[train.shape[0]:].to_pickle(ROOT_PATH+'/test_data2.pkl')
    # data.iloc[:train.shape[0]].to_csv(ROOT_PATH + '/allData2.csv',index=False)
    # data.iloc[train.shape[0]:].to_csv(ROOT_PATH + '/test_data2.csv',index=False)
    # return

    '''
    读取处理好的文件
    '''
    train=pd.read_pickle(ROOT_PATH+'/allData2.pkl')
    test=pd.read_pickle(ROOT_PATH+'/test_data2.pkl')
    # train = pd.read_csv(ROOT_PATH + '/allData2.csv')
    # #train = train.sample(frac=0.2, random_state=42).reset_index(drop=True)
    # test = pd.read_csv(ROOT_PATH + '/test_data2.csv',nrows=10)
    print('load data finish...')
    data = pd.concat((train, test)).reset_index(drop=True)



    videoplayseconds = ['videoplayseconds']
    feedEmbedding_features=[]
    for i in range(64):
        feedEmbedding_features.append('feedembedding_%s'%i)
    dense_features=videoplayseconds+feedEmbedding_features
    sparse_features = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']

    data[sparse_features] = data[sparse_features].fillna(0)
    data[dense_features] = data[dense_features].fillna(0)

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        #重新编号
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[videoplayseconds] = mms.fit_transform(data[videoplayseconds])

    #xxFeat参数：'name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'dtype', 'embedding_name','group_name'
    # sparse features: ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id']
    # dense features: ['videoplayseconds']
    dnn_feature_columns = [SparseFeat(feat, data[feat].nunique(),128)
                                for feat in sparse_features] + [DenseFeat(feat, 1, 1)
                                                                for feat in dense_features]
    linear_feature_columns = [SparseFeat(feat, data[feat].nunique(),128)
                                for feat in sparse_features] + [DenseFeat(feat, 1, 1)
                                                                for feat in dense_features]
    # userid 20000 66
    # feedid 99171 102
    # authorid 18623 66
    # bgm_song_id 23738 72
    # bgm_singer_id 16601 66
    # print('dnn_feature_columns',dnn_feature_columns )
    # print('linear_feature_columns',linear_feature_columns )
    #名字重复了啊
    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model
    train, test = data.iloc[:train.shape[0]].reset_index(drop=True), data.iloc[train.shape[0]:].reset_index(drop=True)
    offlineTrain=train[train['date_']<14].reset_index(drop=True)
    offlineValid=train[train['date_']==14].reset_index(drop=True)
    #onlineTrain=train
    # print('feature_names',feature_names) 
    offlineTrain_input = {name: offlineTrain[name] for name in feature_names}
    offlineValid_input = {name: offlineValid[name] for name in feature_names}


    data=0
    train=0
    test=0

    train_textDict={}
    valid_textDict={}
    train_textDict['manual_keyword_list']=offlineTrain['manual_keyword_list'].values.tolist()
    train_textDict['machine_keyword_list']=offlineTrain['machine_keyword_list'].values.tolist()
    train_textDict['itsc_keyword_list']=offlineTrain['itsc_keyword_list'].values.tolist()
    train_textDict['manual_tag_list']=offlineTrain['manual_tag_list'].values.tolist()
    train_textDict['machine_tag_list']=offlineTrain['machine_tag_list'].values.tolist()
    train_textDict['itsc_tag_list']=offlineTrain['itsc_tag_list'].values.tolist()
    
    valid_textDict['manual_keyword_list']=offlineValid['manual_keyword_list'].values.tolist()
    valid_textDict['machine_keyword_list']=offlineValid['machine_keyword_list'].values.tolist()
    valid_textDict['itsc_keyword_list']=offlineValid['itsc_keyword_list'].values.tolist()
    valid_textDict['manual_tag_list']=offlineValid['manual_tag_list'].values.tolist()
    valid_textDict['machine_tag_list']=offlineValid['machine_tag_list'].values.tolist()
    valid_textDict['itsc_tag_list']=offlineValid['itsc_tag_list'].values.tolist()
    other={'train_textDict':train_textDict,
            'valid_textDict':valid_textDict}
def train(rate=0):
        #test_model_input = {name: test[name] for name in feature_names}

        # 4.Define Model,train,predict and evaluate
        device = 'gpu'
        use_cuda = True
        if use_cuda and torch.cuda.is_available():
            print('cuda ready...')
            device = 'cuda:0'
        # print('linear_feature_columns',linear_feature_columns)
        # print('dnn_feature_columns',dnn_feature_columns)
        # break
        
        model = MyDeepFM(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,
                       task='binary',use_fm=True,device=device,rate=rate)
        #3.6e-2 3.62e-2 4.2e-2 3.8e-2
        optimizer = torch.optim.Adagrad(model.parameters(), lr=rate)
        ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        model.compile(optimizer,"binary_crossentropy",metrics=['binary_crossentropy'])
        #当verbose=1时，带进度条的输出日志信息
        print('begin train...')
        history = model.fit(offlineTrain_input,offlineTrain[actions].values, batch_size=512, epochs=3, verbose=1,
                            validation_data=(offlineValid_input,offlineValid[actions].values),other=other)
        # pred_ans = model.predict(test_model_input, 512)
        # submit[action] = pred_ans
        torch.cuda.empty_cache()
    # weight_uAuc,uAUC=uAUC2(offlineValid[USER_ACTION].values.reshape(-1,4),submit[USER_ACTION].values.reshape(-1,4),offlineValid['userid'].values.flatten())
    # print("weight_uAuc:",weight_uAuc,"uAuc0:",uAUC[0],"uAuc1:",uAUC[1],"uAuc2:",uAUC[2],"uAuc3:",uAUC[3])
    # 保存提交文件
    #submit.to_csv(ROOT_PATH+"/submit/submit_base_deepfm2.csv", index=False)
if __name__ == "__main__":


    #ratelist=[1e-2,2e-2,3e-2,3.5e-2,2.5e-2,4e-2,5e-2,6e-2,7e-2]
    ratelist=[1.6e-2,1.8e-2,2e-2,2.2e-2,2.4e-2,3e-2,4e-2]
    for rate in ratelist:
        record[str(rate)]={}
        for i in range(3):
            record[str(rate)][str(i)]={}
            record[str(rate)][str(i)]['uAuc']=0

    for rate in ratelist:
        train(rate)
    
    with open('/home/shiyunxiao/deepCTR/wechat_big_data_baseline_pytorch/record/record2.txt','w') as f:
        for rate in ratelist:
            f.write(str(rate))
            f.write('\n')
            for i in range(3):
                f.write(str(i))
                f.write('\n')
                f.write(record[str(rate)][str(i)]['uAuc'])
                f.write('\n')