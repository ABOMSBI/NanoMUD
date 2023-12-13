import numpy as np
import pandas as pd
import os, csv, re, sys, argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torcheval.metrics import MeanSquaredError
from torch.utils.data import Dataset, DataLoader, random_split

def sampling_probs(df,times):
    num = 200
    prob_dist = np.zeros((times,5))
    prob_dist = pd.DataFrame(prob_dist, columns=['prob_mean','prob_median','prob_1q','prob_3q','prob_sd'])
    for i in range(times):
        if df.shape[0] > num:
            tmp = df.sample(n = num, axis = 0)
        else:
            tmp = df.sample(n = num, axis = 0, replace=True)

        prob_dist.iloc[i,:] = [tmp['prob'].mean(),tmp['prob'].median(),tmp['prob'].quantile(0.25),
                               tmp['prob'].quantile(0.75),tmp['prob'].std()]
    return prob_dist

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

def predict(test_loader, model, device, has_label = True):
    model.eval()
    if has_label ==True:
        preds = []
        true_y = []
        for i, batch in enumerate(tqdm(test_loader)):
            features, labels = batch
            true_y.append(labels.detach())
            features = features.to(device)
            with torch.no_grad():
                pred = model(features)
                preds.append(pred.detach().cpu())
        preds = torch.cat(preds, dim=0)
        true_y = torch.cat(true_y,dim=0)
        metric = MeanSquaredError(device=device)
        metric.update(preds,true_y)
        mse = metric.compute().item()
        return preds.numpy(), mse
    else:
        preds = []
        for i, batch in enumerate(tqdm(test_loader)):
            features = batch.to(device)
            with torch.no_grad():
                pred = model(features)
                preds.append(pred.detach().cpu())
        preds = torch.cat(preds, dim=0)
        return preds.numpy()

def predicted_rate(x):
    return len(x[x>0.5])/len(x)

def capping(x):
    if x > 1:
        return 1
    elif x < 0:
        return 0
    else:
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='calibrate site predicted mod rate')
    parser.add_argument('-i', '--input', required=True, help='predicted probs')
    parser.add_argument('-o', '--output', required=True, help='calbirated mod rate')
    parser.add_argument('--device', required=True, help='GPU device')
    parser.add_argument('--model', required=True, help='model dir')
    args = parser.parse_args(sys.argv[1:])

    global FLAGS
    FLAGS = args

    probs = pd.read_csv(FLAGS.input)
    mod_rate = probs.groupby(['chrom','chrom_pos','strand','motif'],as_index=False).agg({'read_id':'count',
                                                                                               'prob':predicted_rate})
    mod_rate.columns = ['chrom','chrom_pos','strand','motif','coverage','predicted_rate']
    mod_rate['calibrated_rate'] = pd.NA
    for i in range(mod_rate.shape[0]):
        chrom = mod_rate.iloc[i,0]
        chrom_pos = mod_rate.iloc[i,1]
        strand = mod_rate.iloc[i,2]
        motif = mod_rate.iloc[i,3]
        tmp = probs.loc[(probs['chrom']==chrom)&(probs['chrom_pos']==chrom_pos)&(probs['strand']==strand)]
        prob_dist = sampling_probs(tmp,times=10)
        config = {'device': FLAGS.device,
                  'model_path': FLAGS.model + '/' + motif + '.ckpt'}
        model = regression_Model(input_dim=5).to(config['device'])
        test_loader = get_Dataloader(prob_dist,training=False,has_label=False)
        model.load_state_dict(torch.load(config['model_path']))

        mod_rate.iloc[i,6] = capping(np.mean(predict(test_loader,model,config['device'],has_label=False)))

    mod_rate.to_csv(FLAGS.output, index = 0)