# -*- coding: utf-8 -*-
"""autoencoder.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oJ1Fg7geHRQcMHIpQtZYni1DLhJKWcyX
"""

# Commented out IPython magic to ensure Python compatibility.
# %cd ../

# Commented out IPython magic to ensure Python compatibility.
# %cd ./content/drive/MyDrive

!pip install flwr

#Autoencoder
import torch
from torch                  import device, no_grad
from torch.cuda             import is_available
from torch.nn               import Linear, Module, MSELoss, ReLU, Sequential, Sigmoid, ELU
from torch.optim            import Adam
from torch.utils.data       import Dataset, DataLoader, SubsetRandomSampler, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import flwr as fl
from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar

data = pd.read_csv("./Autoencoder/creditcard_csv.csv")

data.head()

#data = data.drop(['Time','Amount'], axis=1)
str_map = {"'0'": 0, "'1'": 1}
data['Class'] = data['Class'].map(lambda x: str_map[x])

data.head()

class CustomDataset(Dataset):
    def __init__(self,data, label_column='Class',transform=None):
        self.data = data
        self.features = torch.tensor(self.data.drop(columns=[label_column]).values, dtype=torch.float32)
        self.labels = torch.tensor(self.data[label_column].values, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]

        sample = {"features": feature, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample

class NormalizeStandard(object):
    def __init__(self):
        self.standard_scaler = StandardScaler()

    def __call__(self, sample):
        feature, label = sample['features'], sample['label']
        feature = torch.tensor(self.standard_scaler.fit_transform(feature.reshape(1, -1)), dtype=torch.float32).squeeze()

        return {"features": feature, "label": label}

#Federated splitting between clients
def prepare_dataset(train_dataset,test_dataset,num_part:int,  batch_size: int):
    trainloader = []
    testloader = []

    dataset_len = len(train_dataset)

    # Create a list of partition lengths
    partion_Len = [dataset_len // num_part] * num_part

    # Add any remaining elements to the last partition
    partion_Len[-1] += dataset_len % num_part

    # Split the dataset into multiple smaller datasets
    trainsets = random_split(train_dataset, partion_Len)

    for train in trainsets:

      trainloader.append(DataLoader(train,batch_size=batch_size,shuffle=True))

    test_len = len(test_dataset)

    partion = [test_len // num_part] * num_part
    partion[-1] += test_len % num_part

    testsets = random_split(test_dataset,partion)

    for test in testsets:
      testloader.append(DataLoader(test,batch_size=batch_size,shuffle=True))

    return trainloader,testloader


def preprocessing_data(dataset,num_part:int, split:int,batch_size:int):

  trainloader = []
  testloader = []
  label_column_name = "Class"#Label name for bengign

  #Preprocessing
  transform = transforms.Compose([NormalizeStandard()])
  ######
  custom_dataset = CustomDataset(data, label_column=label_column_name,transform=transform)

  all_train_data, test_data = train_test_split(data, test_size=split, stratify=data[label_column_name].values, random_state=42)

  # Filter train data to keep only samples with label 0
  train_data = all_train_data[all_train_data[label_column_name] == 0]

  train_dataset = CustomDataset(train_data, label_column=label_column_name, transform=transform)
  test_dataset = CustomDataset(test_data, label_column=label_column_name, transform=transform)

  trainloader,testloader = prepare_dataset(train_dataset,test_dataset,num_part,batch_size)

  return trainloader,testloader

class Autoencoder(Module):

    @staticmethod
    def get_non_linear(param):
        def get_one(param):

            if param == 'relu':
                return ReLU(inplace=True)
            if param == 'sigmoid':
                return Sigmoid()
            if param == 'elu':
                return ELU()

            return None

        decoder_non = get_one(param[0])
        encoder_non_linearity = get_one(param[1])

        return encoder_non_linearity, decoder_non

    @staticmethod
    def build_layers(sizes, non_linearity=None):
        linears = [Linear(m, n) for m, n in zip(sizes[:-1], sizes[1:])]

        if non_linearity:
            layers = [item for pair in zip(linears, non_linearity) for item in pair]
        else:
            layers = linears

        return Sequential(*layers)

    def __init__(self,
                 input_dimension,
                 encoder_sizes=[16, 8, 4, 2],
                 encoder_non_linearity='relu',
                 decoder_sizes=[],
                 decoder_non_linearity='relu'):

        super(Autoencoder, self).__init__()
        self.input_dimension = input_dimension
        self.encoder_sizes = [input_dimension] + encoder_sizes
        self.decoder_sizes = decoder_sizes if decoder_sizes else encoder_sizes[::-1]

        encoder_non_linearity, decoder_non_linearity = self.get_non_linear([encoder_non_linearity, decoder_non_linearity])

        self.encoder = self.build_layers(self.encoder_sizes, non_linearity=encoder_non_linearity)
        self.decoder = self.build_layers(self.decoder_sizes + [input_dimension], non_linearity=decoder_non_linearity)

        self.encode = True
        self.decode = True

    def forward(self, x):

        if self.encode:
            x = self.encoder(x)

        if self.decode:
            x = self.decoder(x)

        return x

    def n_encoded(self):
        return self.encoder_sizes[-1]

def train(loader,model,optimizer,criterion,device:str):

    Losses  = []
    N = 100
    model.train()
    model.to(dev)

    for epoch in range(N):
        loss = 0
        for data in loader:
            feature,label = data[0].to(dev), data[1].to(dev)
            optimizer.zero_grad()
            train_loss = criterion(features,labels)
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()

        Losses.append(loss / len(loader))
        print(f'epoch : {epoch+1}/{N}, loss = {Losses[-1]:.6f}')

    return Losses

def test(loader,model,device:str,criterion):

  correct, loss = 0,0.0

  model.eval()
  model.to(device)

  with torch.no_grad():

    for data in loader:
      feature,label = data[0].to(dev), data[1].to(dev)
      outputs = model(feature)
      loss += criterion(outputs,label).item()
      _,predicted = torch.max(outputs.data,1)
      correct += (predicted == labels).sum().item()

    accuracy = correct / len(loader.dataset)
    return loss,accuracy

class FlowerClient(fl.client.NumPyClient):

  def __init__(self,trainloader,input_dimension,encoder_non_linearity,decoder_non_linearity):

    super().__init__()

    self.trainloader = trainloader
    self.model = Autoencoder(input_dimension =input_dimension,
                     encoder_non_linearity = encoder_non_linearity,
                     decoder_non_linearity = decoder_non_linearity)

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def set_parameters(self,parameters):

      params_dict = zip(self.model.state_dict().keys(), parameters)

      state_dict = OrderedDict({k: torch.tensor(v) for k,v in params_dict})

      self.model.load_state_dict(state_dict, strict = True)


    def get_parameters(self, config: Dict[str, Scalar]):

      return [val.cpu().numpy() for _,val in self.model.state_dict().items()]

    def fit(self,parameters):

      self.set_parameters(parameters)

      optimizer = Adam(self.model.parameters(),lr = 0.001)
      criterion  = MSELoss()
      print("Training the client")
      #do local training
      train(self.model,self.trainloader,optimizer,criterion,self.device)

      return self.model.get_parameters(), len(self.trainloader), () #returning the updated weights, () is the additonalal information returned energy



def generate_client_fn(ctrainloader,input_dimension,encoder_non_linearity,decoder_non_linearity):

  def client_fn(cid: str):

    return FlowerClient(trainloader=trainloader[int(cid)],input_dimension= input_dimension,encoder_non_linearity=encoder_non_linearity,decoder_non_linearity=decoder_non_linearity)

  return client_fn

#Server side part

#Return the configration to the client
def get_on_fit_config(config: DictConfig):

  def fit_config_fn(server_round: int):
    return {}

return fit_config_fn

#Main file

#loading datasets
trainloader,testloader = preprocessing_data("xyz",10,0.2,64)

for train in trainloader:
  print(f"Train data dimension: {len(train.dataset)}")

for test in testloader:
  print(f"Test data dimension: {len(test.dataset)}")


data_iter = iter(trainloader[0])
first_batch = next(data_iter)
input_dimension = first_batch['features'].shape[1]

print("Shape of the input dimesion : ", input_dimension)

encoder_non_linearity,decoder_non_linearity = Autoencoder.get_non_linear(['relu','relu'])


client_fn = generate_client_fn(trainloader,input_dimension,encoder_non_linearity,decoder_non_linearity)


strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.0,  # in simulation, since all clients are available at all times, we can just use `min_fit_clients` to control exactly how many clients we want to involve during fit
        min_fit_clients=2,  # number of clients to sample for fit()
        fraction_evaluate=0.0,  # similar to fraction_fit, we don't need to use this argument.
        min_evaluate_clients=1,  # number of clients to sample for evaluate()
        min_available_clients=10,  # total clients in the simulation
        on_fit_config_fn=get_on_fit_config(
            cfg.config_fit
        ),  # a function to execute to obtain the configuration to send to the clients during fit()
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),
    )  # a function to run on the server side to evaluate the global model.

