# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="977" height="858" alt="image" src="https://github.com/user-attachments/assets/61c686c7-0867-4176-aee5-50172758eb16" />

## DESIGN STEPS
STEP 1:
Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding, and encode the target class (Segmentation).

STEP 2:
Split the dataset into training and testing sets, then normalize the input features using StandardScaler for better neural network performance.

STEP 3:
Convert the scaled training and testing data into PyTorch tensors and create DataLoader objects for batch-wise training and evaluation.

STEP 4:
Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, ending with an output layer for multi-class classification.

STEP 5:
Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, loss calculation, backpropagation, and weight updates over multiple epochs.

STEP 6:
Evaluate the trained model on test data using accuracy, confusion matrix, and classification report, and perform prediction on a sample input.
## PROGRAM

### Name: THARUN V

### Register Number: 212224230290

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from torch.utils.data import TensorDataset,DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("customers.csv")
df
df=df.drop(columns=["ID"])
df
df.columns
df.fillna({"Work_Experience":0,"Family_Size":df["Family_Size"].median()},inplace=True)
df
cat_columns=['Gender','Ever_Married', 'Graduated', 'Profession',
    'Spending_Score', 'Var_1']
for col in cat_columns:
    df[col]=LabelEncoder().fit_transform(df[col])
lbe=LabelEncoder()
df["Segmentation"]=lbe.fit_transform(df["Segmentation"])
df
x=df.drop(columns="Segmentation")
y=df["Segmentation"].values
xt,xst,yt,yst=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
xt=scaler.fit_transform(xt)
xst=scaler.transform(xst)

xt=torch.FloatTensor(xt)
xst=torch.FloatTensor(xst)
yt=torch.FloatTensor(yt)
yst=torch.FloatTensor(yst)

tr=TensorDataset(xt,yt)
tst=TensorDataset(xst,yst)
trl=DataLoader(tr,batch_size=16,shuffle=True)
tstl=DataLoader(tst,batch_size=16)

class classifier1(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.l1=nn.Linear(input_size,32)
        self.l2=nn.Linear(32,16)
        self.l3=nn.Linear(16,8)
        self.l4=nn.Linear(8,4)
    def forward(self,x):
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x=F.relu(self.l3(x))
        x=self.l4(x)
        return x


model=classifier1(input_size=xt.shape[1])
criterion=nn.CrossEntropyLoss()
op=optim.Adam(model.parameters(),lr=0.001)

epochs=100
for i in range(epochs):
    for a,b in trl:
        op.zero_grad()
        pred=model(a)
        loss=criterion(pred,b.long())
        loss.backward()
        op.step()
    if i%10==0:
        print(f"Loss:{i}/{epochs}",loss.item())


pre=[]
act=[]
with torch.no_grad():
    output=model(xst)
    _,predicted=torch.max(output,1)
    pre.extend(predicted.numpy())
    act.extend(yst.numpy())
accuracy=accuracy_score(act,pre)
conf_matrix=confusion_matrix(act,pre)
cl_report=classification_report(act,pre,target_names=['A','B','C','D'])
print("Accuracy:",accuracy)
print("confusion_matrix:\n",conf_matrix)
print("classification_report:\n",cl_report)
import seaborn as sns
import matplotlib.pyplot as plt
xl=['A','B','C','D']
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=xl, yticklabels=xl,fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()
```

### Dataset Information
<img width="1130" height="445" alt="image" src="https://github.com/user-attachments/assets/a02db79b-4976-4454-b51f-8c76c905b88c" />

### OUTPUT
<img width="1055" height="442" alt="image" src="https://github.com/user-attachments/assets/3f3cd39d-9036-4f61-8ca4-1e477652e091" />

## Confusion Matrix
<img width="621" height="515" alt="image" src="https://github.com/user-attachments/assets/9dca4404-a0a8-49fb-92b6-7e576cd68541" />


## Classification Report
<img width="497" height="218" alt="image" src="https://github.com/user-attachments/assets/429204c2-cb2e-450d-ac65-26965c12b52a" />

### New Sample Data Prediction
<img width="302" height="118" alt="image" src="https://github.com/user-attachments/assets/9e8065fd-aba6-46ba-bc76-a8fa405c3260" />

## RESULT
This program has been executed successfully.
