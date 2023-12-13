import math, joblib, os, csv, re, argparse
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from torcheval.metrics import BinaryAUROC

def same_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# data preparation
class myDataset(Dataset):
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.LongTensor(y)
        self.x = torch.Tensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return self.x.shape[0]


def get_Dataloader(df, scaler = None, training = True, has_label = True):
    if training == True and has_label == True:
        scaler = StandardScaler()
        df.iloc[:, 6:df.shape[1] - 1] = scaler.fit_transform(df.iloc[:, 6:df.shape[1] - 1])
        dataset = myDataset(np.array(df.iloc[:, 6:df.shape[1] - 1]), np.array(df['label']))
        train_set, val_set = random_split(dataset, (0.7, 0.3))
        train_loader = DataLoader(train_set, batch_size=500, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=500, shuffle=False)
        return (train_loader, train_set, val_loader, val_set, scaler)
    elif training == False and has_label == True:
        df.iloc[:, 6:21] = scaler.transform(df.iloc[:, 6:21])
        test_set = myDataset(np.array(df.iloc[:, 6:21]), np.array(df['label']))
        test_loader = DataLoader(test_set, batch_size=500, shuffle=False)
        return (test_loader, test_set)
    else:
        df.iloc[:, 6:21] = scaler.transform(df.iloc[:, 6:21])
        test_set = myDataset(np.array(df.iloc[:, 6:21]))
        test_loader = DataLoader(test_set, batch_size=500, shuffle=False)
        return (test_loader, test_set)

class biLSTM_model(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(biLSTM_model, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 2, bidirectional=True)
        self.fcnn = nn.Sequential(
            nn.Linear(hidden_dim*2,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,2)
        )
    def forward(self, event):
        out, (h_n, c_n) = self.lstm(event)
        x = self.fcnn(out)
        return x

def predict(test_loader, model, device, return_auc = True):
    model.eval()
    if return_auc ==True:
        preds = []
        true_y = []
        for i, batch in enumerate(tqdm(test_loader)):
            features, labels = batch
            true_y.append(labels.detach())
            features = features.to(device)
            with torch.no_grad():
                pred = model(features)
                pred = nn.functional.softmax(pred, dim=1)
                preds.append(pred.detach().cpu())
        preds = torch.cat(preds, dim=0)
        true_y = torch.cat(true_y,dim=0)
        metric = BinaryAUROC(device=device)
        metric.update(preds[:, 1], true_y)
        auc = metric.compute().item()
        return preds[:,1], auc
    else:
        preds = []
        for i, batch in enumerate(tqdm(test_loader)):
            features = batch.to(device)
            with torch.no_grad():
                pred = model(features)
                pred = nn.functional.softmax(pred, dim=1)
                preds.append(pred.detach().cpu())
        preds = torch.cat(preds, dim=0)
        return preds[:,1]

def concat_df(path):
    csvs = os.listdir(path)
    df = pd.DataFrame()
    for csv in csvs:
        if re.search(r'\.csv', csv):
            csv_dir = os.path.join(path, csv)
            tmp = pd.read_csv(csv_dir)
            df = pd.concat([df,tmp],axis=0)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read-level modified probability prediction')
    parser.add_argument('-i', '--input', required=True, help='./tmp feature dir')
    parser.add_argument('-o', '--output', required=True, help='output file name')
    parser.add_argument('--device', required=True, help='GPU device')
    parser.add_argument('--model', required=True, help='model dir')
    parser.add_argument('--scaler', required=True, help='scaler dir')
    args = parser.parse_args(sys.argv[1:])

    global FLAGS
    FLAGS = args

    testing = concat_df(FLAGS.input)
    testing = testing.rename(columns = {'0':'chrom','1':'chrom_pos','2':'strand',
                                        '3':'read_id','4':'read_pos','5':'motif'})
    motifs = testing['motif'].unique()

    testing['prob'] = pd.NA
    for motif in motifs:
        try:
            config = {'device': FLAGS.device,
                      'scaler_path': FLAGS.scaler + '/' + motif + '.joblib',
                      'model_path': FLAGS.model + '/' + motif + '.ckpt'}

            model = biLSTM_model(input_dim=15,hidden_dim=20).to(config['device'])
            model.load_state_dict(torch.load(config['model_path']))
            scaler = joblib.load(config['scaler_path'])

            test_df = testing.loc[testing['motif'] == motif,:]
            test_loader, test_set = get_Dataloader(test_df, scaler, training=False, has_label=False)
            testing.loc[testing['motif'] == motif,'prob'] = predict(test_loader,model,config['device'],return_auc=False)
        except Exception as e:
            print(str(e))
            continue

    testing = testing.loc[:,['chrom','chrom_pos','strand','read_id','read_pos','motif','prob']]
    testing.to_csv(FLAGS.output, index = 0)





