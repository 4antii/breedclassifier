# Breed Classifier
Cats and Dogs breed classification using telegram-bot-api. 

This project was made as final task of Machine Learning course during SPBU Master program.

## Dataset 
For training we used "Cats and Dogs Breeds Classification Oxford Dataset". You can find all information about dataset with the link: 

https://www.kaggle.com/datasets/zippyz/cats-and-dogs-breeds-classification-oxford-dataset

## Model 
The classification model is ResNet18 generic implementation from Pytorch. Transfer learning technique was used to adapt pretrained weight to selected dataset. You can download ResNet checkpoint (resnet18_s2_700.pt) and load our trained model using following commands:
```
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model.fc = nn.Sequential(torch.nn.Linear(512, 37), nn.Softmax())
model.load_state_dict(torch.load('./USE_YOUR_PATH/resnet18_s2_700.pt'))
```
## Requirements 
To install all required dependencies simply run the command in your terminal:

```
pip install -r requirements.txt
```

## How to run Telgram bot 
Run the following command to host the breed classification bot:

```
#!/usr/bin/env python3 bot.py
```
