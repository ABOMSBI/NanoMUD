import math, joblib, os, csv, re, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from torcheval.metrics import BinaryAUROC
from sklearn.model_selection import train_test_split

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

# training functions
def trainer(train_loader, val_loader, train_set, val_set,model, config):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    best_loss = 100
    best_model = model
    for epoch in range(config['n_epochs']):
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
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

            _, train_pred = torch.max(outputs, 1)
            train_acc += (train_pred.detach() == labels.detach()).sum().item()
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

                _, val_pred = torch.max(outputs, 1)
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
                val_loss += loss.item()

            if val_loss < best_loss:
                best_loss = val_loss
                best_model = model
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
            epoch + 1, config['n_epochs'], train_acc/len(train_set),
            train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(val_loader)
        ))
    return (best_model)

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


# concat feature files from ./tmp
def concat_df(path):
    csvs = os.listdir(path)
    df = pd.DataFrame()
    for csv in csvs:
        if re.search(r'\.csv', csv):
            csv_dir = os.path.join(path, csv)
            tmp = pd.read_csv(csv_dir)
            df = pd.concat([df,tmp],axis=0)
    return df

# prepare training and testing dataset
neg = concat_df(r'/home/yuxin/Desktop/psi/normal/tmp/')
neg['label'] = 0
pos = concat_df(r'/home/yuxin/Desktop/psi/psi/tmp2/')
pos['label'] = 1
data = pd.concat([pos,neg],axis=0)
del pos
del neg

data = data.rename(columns = {'0':'chrom','1':'chrom_pos',
                              '2':'strand','3':'read_id','4':'read_pos','5':'motif'})

training,testing = train_test_split(data,train_size=0.7,test_size=0.3)
training.to_csv('/home/yuxin/Desktop/psi/training.csv',index = 0)
testing.to_csv('/home/yuxin/Desktop/psi/testing.csv',index = 0)

del data
del training
del testing

same_seed(5201314)
os.system('mkdir -p /home/yuxin/Desktop/psi/biLSTM_model')
os.system('mkdir -p /home/yuxin/Desktop/psi/scaler')

training = pd.read_csv('/home/yuxin/Desktop/psi/training.csv')
testing = pd.read_csv('/home/share/yuxin/new_qc/testing.csv')

train_perf = pd.DataFrame(columns=['motif','auc'])
train_perf['motif'] = training['motif'].unique()

for motif in motifs:
    try:
        config = {'device': 'cuda:0',
                  'n_epochs': 300,
                  'batch_size': 500,
                  'learning_rate': 0.001,
                  'save_path': '/home/yuxin/Desktop/psi/biLSTM_model/' + motif + '.ckpt'}
        model = biLSTM_model(input_dim=15,hidden_dim=20).to(config['device'])
        train_df = training.loc[training['motif']==motif,:]
        train_loader, train_set, val_loader, val_set, scaler = get_Dataloader(train_df, scaler=None, training = True, has_label=True)
        joblib.dump(scaler, '/home/yuxin/Desktop/psi/scaler/' + motif + '.joblib')
        best_model = trainer(train_loader, val_loader, train_set, val_set, model, config)
        torch.save(best_model.state_dict(), config['save_path'])
        preds, auc = predict(val_loader, best_model, config['device'], return_auc= True)
        train_perf.loc[train_perf['motif']==motif,'auc'] = auc
        print('save ' + str(motif) + ' model with auc: ' + str(auc))

    except Exception as e:
        print(str(e))
        continue
train_perf.to_csv('/home/yuxin/Desktop/psi/train_perf.csv',index=0)

# testing
def predict_by_biLSTM(testing, has_label = True):
    motifs = testing['motif'].unique()

    if has_label ==True:
        test_perf = pd.DataFrame(columns=['motif','auc'])
        test_perf['motif'] = motifs

    testing['prob'] = pd.NA
    for motif in motifs:
        try:
            config = {'device': 'cuda:0',
                      'scaler_path': '/home/yuxin/Desktop/psi/scaler/' + motif + '.joblib',
                      'model_path': '/home/yuxin/Desktop/psi/biLSTM_model/' + motif + '.ckpt'}

            model = biLSTM_model(input_dim=15,hidden_dim=20).to(config['device'])
            model.load_state_dict(torch.load(config['model_path']))
            scaler = joblib.load(config['scaler_path'])

            test_df = testing.loc[testing['motif'] == motif,:]
            test_loader, test_set = get_Dataloader(test_df, scaler, training=False, has_label=has_label)
            if has_label == True:
                testing.loc[testing['motif'] == motif,'prob'],auc = predict(test_loader,model,config['device'])
                test_perf.loc[test_perf['motif']==motif,'auc'] = auc
                print('test auc of ' + str(motif) + ' is ' + str(auc))
            else:
                testing.loc[testing['motif'] == motif,'prob'] = predict(test_loader,model,config['device'],return_auc=False)
        except Exception as e:
            print(str(e))
            continue
    if has_label ==True:
        testing = testing.loc[:,['chrom','chrom_pos','strand','read_id','read_pos','motif','label','prob']]
        return testing, test_perf
    else:
        testing = testing.loc[:,['chrom','chrom_pos','strand','read_id','read_pos','motif','prob']]
        return testing

testing = pd.read_csv('/home/yuxin/Desktop/psi/testing.csv')
testing, test_perf = predict_by_biLSTM(testing,has_label=True)
testing.to_csv('/home/yuxin/Desktop/psi/biLSTM_test_pred.csv', index = 0)
test_perf.to_csv('/home/yuxin/Desktop/psi/test_perf.csv', index = 0)