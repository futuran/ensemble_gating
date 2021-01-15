import torch
import torch.nn as nn
import torchtext
import torch.nn.functional as F
import torch.optim as optim

import time
import pprint



class Net(nn.Module):
    def __init__(self,device):
        super(Net, self).__init__()
        self.preluw = torch.tensor([0.25],device=device)

        self.fc1 = nn.Linear(600,600)
        self.fc2 = nn.Linear(600,150)
        self.fc3 = nn.Linear(150,40)
        self.fc4 = nn.Linear(40,10)
        self.fc5 = nn.Linear(10,2)
        self.sm  = nn.Softmax(dim=1)

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self,x):
        #print(x)

        #time.sleep(10)
        x = F.prelu(self.fc1(x),self.preluw)
        x = F.prelu(self.fc2(x),self.preluw)
        x = self.dropout1(x)
        x = F.prelu(self.fc3(x),self.preluw)
        x = F.prelu(self.fc4(x),self.preluw)
        x = self.dropout2(x)
        x = F.prelu(self.fc5(x),self.preluw)

        x = self.sm(x)
        return x



def loader():
    '''
    the format of dataset are bellow.
    Hello<\space>world<\space>.<\tab><\n>
    cat output.train1.fr | tr "\n" "\\t\n" > output.train1.tsv
    '''
    
    print("Loading Data...")
    max_length = 100
    TEXT = torchtext.data.Field(sequential=True, 
                    use_vocab=True, lower=False, 
                    include_lengths=True,batch_first=True, 
                    fix_length=max_length)
    
    LABEL = torchtext.data.Field(sequential=False, use_vocab=False)
    
    train_ds, val_ds, test_ds = torchtext.data.TabularDataset.splits(path='./',
                                                    train='../10.ensemble/train_balanced_shuf.tsv', 
                                                    validation='../10.ensemble/valid.alltab.tsv', 
                                                    test='../10.ensemble/test.alltab.tsv', 
                                                    format='tsv', fields=[('Input',TEXT),('Baseline',TEXT),('Nfr',TEXT),('Label',LABEL)])


    TEXT.build_vocab(train_ds, min_freq=1)
    #for example in train_ds:
    #print(*test_ds[0].Input)
    #print(TEXT.vocab.freqs)

    return train_ds, val_ds, test_ds

def train(train_ds, val_ds, test_ds):

    print('Building Iterator...')
    train_dl = torchtext.data.Iterator(train_ds, batch_size=5000, train=True)
    val_dl   = torchtext.data.Iterator(val_ds, batch_size=496, train=False, sort=False)
    test_dl  = torchtext.data.Iterator(test_ds, batch_size=500, train=False, sort=False)
    dls = {'train':train_dl, 'val':val_dl, 'test':test_dl}

    device = 'cuda'
    net =Net(device).to(device)
    print(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)


    num_epoch = 300

    trian_loss_list = []
    trian_acc_list = []
    val_loss_list =[]
    val_acc_list = []

    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epoch):
        epoch_loss = 0
        epoch_acc = 0

        for phase in ['train', 'val', 'test']:
            if phase == 'train':
                net.train()
            else:
                net.eval()


            for batch in dls[phase]:
                input_sens = batch.Input[0].to(device).to(dtype=torch.float32)
                baseline_sens = batch.Baseline[0].to(device).to(dtype=torch.float32)
                nfr_sens = batch.Nfr[0].to(device).to(dtype=torch.float32)

                input_bool = (input_sens-1).clone().detach().to(dtype=torch.bool).to(dtype=torch.float32)
                baseline_bool = (baseline_sens-1).clone().detach().to(dtype=torch.bool).to(dtype=torch.float32)
                nfr_bool = (nfr_sens-1).clone().detach().to(dtype=torch.bool).to(dtype=torch.float32)
                
                inputs = torch.cat([input_sens, baseline_sens, nfr_sens, input_bool, baseline_bool, nfr_bool],dim=1)

                inputs.requires_grad=True

                labels = batch.Label.to(device)

                optimizer.zero_grad()   # initialized optimaizer
                outputs = net(inputs)
                #print(outputs)
                _, preds = torch.max(outputs, 1)
                #print(f'preds={preds}')
                #print(preds.dtype)
                #print(labels)
                loss = criterion(outputs, labels)
                #print(loss)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)
                epoch_acc  += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dls[phase].dataset)
            epoch_acc = epoch_acc.double() / len(dls[phase].dataset)
            print('Epoch {}/{} | {:^5} | Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, num_epoch,phase, epoch_loss, epoch_acc))

            if epoch%10 == 0:
                torch.save(net.state_dict(), 'epoch' + str(epoch) + '.ckpt')

def suiron(train_ds, val_ds, test_ds):
    device = 'cuda'
    net = Net(device).to(device)
    net.load_state_dict(torch.load('epoch100.ckpt'))
    print(net)

    print('Building Iterator...')
    train_dl = torchtext.data.Iterator(train_ds, batch_size=5000, train=True)
    val_dl   = torchtext.data.Iterator(val_ds, batch_size=496, train=False, sort=False)
    test_dl  = torchtext.data.Iterator(test_ds, batch_size=1000, train=False, sort=False)
    dls = {'train':train_dl, 'val':val_dl, 'test':test_dl}


    net.eval()

    for batch in dls['test']:
        input_sens = batch.Input[0].to(device).to(dtype=torch.float32)
        baseline_sens = batch.Baseline[0].to(device).to(dtype=torch.float32)
        nfr_sens = batch.Nfr[0].to(device).to(dtype=torch.float32)

        input_bool = (input_sens-1).clone().detach().to(dtype=torch.bool).to(dtype=torch.float32)
        baseline_bool = (baseline_sens-1).clone().detach().to(dtype=torch.bool).to(dtype=torch.float32)
        nfr_bool = (nfr_sens-1).clone().detach().to(dtype=torch.bool).to(dtype=torch.float32)
        
        inputs = torch.cat([input_sens, baseline_sens, nfr_sens, input_bool, baseline_bool, nfr_bool],dim=1)

        inputs.requires_grad=True

        labels = batch.Label.to(device)

        #optimizer.zero_grad()   # initialized optimaizer
        outputs = net(inputs)
        #print(outputs)
        #outputs[:,1] = outputs[:,1]*100
        #print(outputs[:,0])
        _, preds = torch.max(outputs, 1)

        for i, pred in enumerate(preds):
            pred = pred.item()
            if pred == 0:
                print(*test_ds[i].Baseline)
            elif pred == 1:
                print(*test_ds[i].Nfr)


        #print(torch.sum(preds == labels.data))


def main():
    train_ds, val_ds, test_ds = loader()
    train(train_ds, val_ds, test_ds)

    #suiron(train_ds, val_ds, test_ds)

main()