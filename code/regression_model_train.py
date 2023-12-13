import numpy as np
import pandas as pd
import os, csv, re
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torcheval.metrics import MeanSquaredError

# prepare training sample
train_pred = pd.read_csv('/home/yuxin/Desktop/psi/biLSTM_test_pred.csv')
motifs = train_pred['motif'].unique()

def get_prob_dist(df,motif,times):
    num = 1000 # sample size
    mod_ratios = np.random.rand(times).tolist()
    mod = df.loc[df['label']==1,:]
    unmod = df.loc[df['label']==0,:]
    prob_dist = np.zeros((times,7))
    prob_dist = pd.DataFrame(prob_dist, columns=['prob_mean','prob_median','prob_1q','prob_3q','prob_sd','mod_rate','predicted_rate'])
    for i in range(times):
        mod_tmp = mod.sample(n = int(num*mod_ratios[i]), axis = 0,replace=True)
        unmod_tmp = unmod.sample(n = num - int(num*mod_ratios[i]), axis = 0,replace=True)
        tmp = pd.concat([mod_tmp,unmod_tmp],axis=0)
        predicted_rate = tmp.loc[tmp['prob'] > 0.5,:].shape[0]/tmp.shape[0]
        prob_dist.iloc[i,:] = [tmp['prob'].mean(),tmp['prob'].median(),tmp['prob'].quantile(0.25),
                               tmp['prob'].quantile(0.75),tmp['prob'].std(),mod_ratios[i],predicted_rate]
    prob_dist['motif'] = motif
    return prob_dist

prob_dists = pd.DataFrame()
for motif in motifs:
    df = train_pred.loc[train_pred['motif']==motif,:]
    prob_dist = get_prob_dist(df,motif,10000)
    prob_dists = pd.concat([prob_dists,prob_dist],axis=0)

prob_dists.to_csv('/home/yuxin/Desktop/psi/regression_train.csv',index = 0)

prob_dists = pd.DataFrame()
for motif in motifs:
    df = train_pred.loc[train_pred['motif']==motif,:]
    prob_dist = get_prob_dist(df,motif,2000)
    prob_dists = pd.concat([prob_dists,prob_dist],axis=0)

prob_dists.to_csv('/home/yuxin/Desktop/psi/regression_test.csv',index = 0)

# training
class myDataset(Dataset):
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

class regression_Model(nn.Module):
    def __init__(self, input_dim):
        super(regression_Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1)
        return x

def get_Dataloader(df,training = True, has_label = True):
    if training == True and has_label==True:
        dataset = myDataset(np.array(df.loc[:,['prob_mean','prob_median','prob_1q','prob_3q','prob_sd']]), np.array(df['mod_rate']))
        train_set, val_set = random_split(dataset, (0.8, 0.2))
        train_loader = DataLoader(train_set, batch_size=500, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=500, shuffle=False)
        return train_loader, val_loader
    elif training == False and has_label==True:
        test_set = myDataset(np.array(df.loc[:,['prob_mean','prob_median','prob_1q','prob_3q','prob_sd']]), np.array(df['mod_rate']))
        test_loader = DataLoader(test_set, batch_size=500, shuffle=False)
        return test_loader
    else:
        dataset = myDataset(np.array(df.loc[:,['prob_mean','prob_median','prob_1q','prob_3q','prob_sd']]))
        data_loader = DataLoader(dataset, batch_size=500, shuffle=False)
        return data_loader

def trainer(train_loader, val_loader,model, config):
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    best_loss = 100
    best_model = model
    for epoch in range(config['n_epochs']):
        model.train()
        train_loss = 0.0
        val_loss = 0.0

        # training
        for i, batch in enumerate(tqdm(train_loader)):
            features, labels = batch
            features = features.to(config['device'])
            labels = labels.to(config['device'])

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # validation
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader)):
                features, labels = batch
                features = features.to(config['device'])
                labels = labels.to(config['device'])
                outputs = model(features)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model
        print('[{:03d}/{:03d}] train Loss: {:3.6f} |  validation loss: {:3.6f}'.format(
            epoch + 1, config['n_epochs'], train_loss/len(train_loader), val_loss/len(val_loader)
        ))
    return (best_model)

training = pd.read_csv('/home/yuxin/Desktop/psi/regression_train.csv')
motifs= training['motif'].unique()
testing = pd.read_csv('/home/yuxin/Desktop/psi/regression_test.csv')

os.system('mkdir -p /home/yuxin/Desktop/psi/regression_model')

train_perf = pd.DataFrame(columns=['motif','mse'])
train_perf['motif'] = training['motif'].unique()

for motif in motifs:
    try:
        config = {'device': 'cuda:0',
                  'n_epochs': 5000,
                  'learning_rate': 0.001,
                  'save_path': '/home/yuxin/Desktop/psi/regression_model/' + motif + '.ckpt'}

        model = regression_Model(input_dim=5).to(config['device'])
        train_df = training.loc[training['motif']==motif,:]
        test_df = testing.loc[testing['motif']==motif,:]
        train_loader, val_loader = get_Dataloader(train_df,training=True)
        test_loader = get_Dataloader(test_df,training=False)
        best_model = trainer(train_loader, val_loader, model, config)
        torch.save(best_model.state_dict(), config['save_path'])
        testing.loc[testing['motif']==motif,'calibrated_rate'], mse = predict(test_loader, best_model, config['device'], has_label=True)
        train_perf.loc[train_perf['motif']==motif,'mse'] = mse
        print(motif + ":" + str(mse))
    except Exception as e:
        print(str(e))
        continue

train_perf.to_csv('/home/yuxin/Desktop/psi/train_perf.csv',index = 0)