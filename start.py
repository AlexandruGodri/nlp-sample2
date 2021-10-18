# !pip install transformers==3
# !pip install torchinfo
# !pip install nltk

import numpy as np
import pandas as pd
import re
import torch
import random
import transformers
import torch.nn as nn
from torchinfo import summary
import matplotlib.pyplot as plt
import nltk
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import AdamW
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import lr_scheduler

nltk.download('stopwords')
nltk.download('punkt')

# specify GPU
device = torch.device("cpu")

# Reading the data
df = pd.read_csv("chat.csv")

# Converting the labels into encodings
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Checking class distribution
print(df['label'].value_counts(normalize = True))

train_text, train_labels = df['text'], df['label']

# Load the DistilBert tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# Import the DistilBert pretrained model
bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Function for removing stopwords
def remove_stopwords(str):
  stop_words = set(stopwords.words('english')) 
  word_tokens = word_tokenize(str)
  filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]  
  return ' '.join(filtered_sentence)

# Remove stop words
train_text = train_text.map(lambda x: remove_stopwords(x))
# Remove punctuatuion
train_text = train_text.map(lambda x: re.sub(r'[\W\s]', '', x))

# Plotting all the messages in order to get the max length
sequence_len = [len(i.split()) for i in train_text]
pd.Series(sequence_len).hist(bins = 10)
# Selecting the max len as 10 (based on the plot)
max_seq_len = 8

# tokenize and encode sequences in the training set
tokens_train = tokenizer(
    train_text.tolist(),
    max_length=max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
)

# for train set
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

#define a batch size
batch_size = 16
# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)
# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)
# DataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

class BERT_Arch(nn.Module):
   def __init__(self, bert):      
       super(BERT_Arch, self).__init__()
       self.bert = bert 
      
       # dropout layer
       self.dropout = nn.Dropout(0.2)
      
       # relu activation function
       self.relu =  nn.ReLU()
       # dense layer
       self.fc1 = nn.Linear(768,512)
       self.fc2 = nn.Linear(512,256)
       self.fc3 = nn.Linear(256,len(np.unique(df['label'])))
       #softmax activation function
       self.softmax = nn.LogSoftmax(dim=1)
       #define the forward pass

   def forward(self, sent_id, mask):
      #pass the inputs to the model  
      cls_hs = self.bert(sent_id, attention_mask=mask)[0][:,0]
      
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      
      x = self.fc2(x)
      x = self.relu(x)
      x = self.dropout(x)
      # output layer
      x = self.fc3(x)
   
      # apply softmax activation
      x = self.softmax(x)
      return x

# freeze all the parameters. This will prevent updating of model weights during fine-tuning.
for param in bert.parameters():
      param.requires_grad = False

model = BERT_Arch(bert)
# push the model to GPU
model = model.to(device)

print(summary(model))

# define the optimizer
optimizer = AdamW(model.parameters(), lr = 1e-3)

#compute the class weights
class_wts = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
print(class_wts)

# convert class weights to tensor
weights= torch.tensor(class_wts,dtype=torch.float)
weights = weights.to(device)
# loss function
cross_entropy = nn.NLLLoss(weight=weights) 

# empty lists to store training and validation loss of each epoch
train_losses = []
# number of training epochs
epochs = 200

# We can also use learning rate scheduler to achieve better results
lr_sch = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

# function to train the model
def train():
  model.train()
  total_loss = 0
  
  # empty list to save model predictions
  total_preds = []
  
  # iterate over batches
  for step, batch in enumerate(train_dataloader):
    # progress update after every 50 batches.
    if step % 50 == 0 and step > 0:
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

    # push the batch to gpu
    batch = [r.to(device) for r in batch] 
    sent_id, mask, labels = batch

    # get model predictions for the current batch
    preds = model(sent_id, mask)

    # compute the loss between actual and predicted values
    loss = cross_entropy(preds, labels)

    # add on to the total loss
    total_loss = total_loss + loss.item()

    # backward pass to calculate the gradients
    loss.backward()

    # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # update parameters
    optimizer.step()

    # clear calculated gradients
    optimizer.zero_grad()
  
    # We are not using learning rate scheduler as of now
    # lr_sch.step()
    preds = preds.detach().cpu().numpy()

    # append the model predictions
    total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)
    
    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds = np.concatenate(total_preds, axis=0)

    #returns the loss and predictions
    return avg_loss, total_preds

for epoch in range(epochs):
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    
    #train model
    train_loss, _ = train()
    
    # append training and validation loss
    train_losses.append(train_loss)

    # it can make the experiment reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'\nTraining Loss: {train_loss:.3f}')

with open("intents.json") as jsonFile:
    data = json.load(jsonFile)
    jsonFile.close()

def get_prediction(str):
  str = re.sub(r'[^a-zA-Z ]+', '', str)
  # Remove stopwords
  test_text = remove_stopwords(str)
  test_text = [re.sub(r'[\W\s]', ' ', test_text)]
  model.eval()
 
  tokens_test_data = tokenizer(
    test_text,
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
  )
  test_seq = torch.tensor(tokens_test_data['input_ids'])
  test_mask = torch.tensor(tokens_test_data['attention_mask'])
 
  preds = None
  with torch.no_grad():
    preds = model(test_seq.to(device), test_mask.to(device))
  preds = preds.detach().cpu().numpy()
  preds = np.argmax(preds, axis = 1)

  # print("Intent Identified: ", le.inverse_transform(preds)[0])
  return le.inverse_transform(preds)[0]

def get_response(message): 
  intent = get_prediction(message)
  for item in data['intents']: 
    if item['tag'] == intent:
      result = random.choice(item['responses'])
      break
  # print(f"Response : {result}")
  return 'Bot: ' + str(result), str(result), str(intent)

print("Ask me anything. Type 'quit' to exit.")
while True:
  sentence = input('You (ask me something): ')
  if sentence == 'quit':
    break
  response_bot, response, intent = get_response(sentence)
  print(response_bot)
  print("Bot: Was this helpful? Please type y/n.")
  helpful = input('You: ')
  if helpful == 'quit':
    break
  if helpful == 'y':
    # should add the question and intent to the training csv
    if not df['text'].isin([sentence]).any():
      with open('chat.csv', 'a') as fd:
        fd.write(sentence + "," + intent + "\n")
        fd.close()
      print("ok")
