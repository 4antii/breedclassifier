from email.mime import image
import torch
import numpy as np
from telegram.ext import Updater, Filters, CommandHandler, MessageHandler
from torchvision import transforms
from torch import nn
from PIL import Image
from io import BytesIO


labels_map = {
    0: "Animal: Cat / Breed: Abyssinian",
    1: "Animal: Cat / Breed: Bengal",
    2: "Animal: Cat / Breed: Birman",
    3: "Animal: Cat / Breed: Bombay",
    4: "Animal: Cat / Breed: British Shorthair",
    5: "Animal: Cat / Breed: Egyptian Mau",
    6: "Animal: Cat / Breed: Maine Coon",
    7: "Animal: Cat / Breed: Persian",
    8: "Animal: Cat / Breed: Ragdoll",
    9: "Animal: Cat / Breed: Russian Blue",
    10: "Animal: Cat / Breed: Siamese",
    11: "Animal: Cat / Breed: Sphynx",
    12: "Animal: Dog / Breed: American Bulldog",
    13: "Animal: Dog / Breed: American Pit Bull Terrier",
    14: "Animal: Dog / Breed: Basset Hound",
    15: "Animal: Dog / Breed: Beagle",
    16: "Animal: Dog / Breed: Boxer",
    17: "Animal: Dog / Breed: Chihuahua",
    18: "Animal: Dog / Breed: English Cocker Spaniel",
    19: "Animal: Dog / Breed: English Setter",
    20: "Animal: Dog / Breed: German Shorthaired",
    21: "Animal: Dog / Breed: Great Pyrenees",
    22: "Animal: Dog / Breed: Havanese",
    23: "Animal: Dog / Breed: Japanese Chin",
    24: "Animal: Dog / Breed: Keeshond",
    25: "Animal: Dog / Breed: Leonberger",
    26: "Animal: Dog / Breed: Miniature Pinscher",
    27: "Animal: Dog / Breed: Newfoundland",
    28: "Animal: Dog / Breed: Pomeranian",
    29: "Animal: Dog / Breed: Pug",
    30: "Animal: Dog / Breed: Saint Bernard",
    31: "Animal: Dog / Breed: Samoyed",
    32: "Animal: Dog / Breed: Scottish Terrier",
    33: "Animal: Dog / Breed: Shiba Inu",
    34: "Animal: Dog / Breed: Staffordshire Bull Terrier",
    35: "Animal: Dog / Breed: Wheaten Terrier",
    36: "Animal: Dog / Breed: Yorkshire Terrier"
}

device = "cuda" if torch.cuda.is_available() else "cpu"


model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18')
model.fc = nn.Sequential(torch.nn.Linear(512, 37), nn.Softmax())
model.load_state_dict(torch.load('./resnet18_s2_700.pt', map_location=torch.device('cpu')))
model = model.to(device)

mean = [0.4803, 0.4503, 0.3951]
std = [0.2627, 0.2583, 0.2669]

def start(updater, context):
	updater.message.reply_text("Welcome to the classification bot!")

def help_(updater, context):
	updater.message.reply_text("Just send the image you want to classify")

def message(updater, context):
	msg = updater.message.text
	print(msg)
	updater.message.reply_text(msg)

def image_handler(update, context):
    model.eval()
    file = context.bot.get_file(update.message.photo[-1].file_id)
    f =  BytesIO(file.download_as_bytearray())
    image = Image.open(f)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean, std)
    ])

    img_tensor = transform(image).float()
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    output = model(img_tensor)
    print(output)
    _, pred = torch.max(output, 1)
    print(pred)
    predicted_breed = labels_map[pred[0].numpy().item()]
    print(predicted_breed)
    update.message.reply_text(predicted_breed)

updater = Updater("5367126523:AAEYaKEMIJmmxWxuSqCgVrM_YBAAEbeYh9E")

dispatcher = updater.dispatcher

dispatcher.add_handler(CommandHandler("start", start))
dispatcher.add_handler(CommandHandler("help", help_))

dispatcher.add_handler(MessageHandler(Filters.text, message))

dispatcher.add_handler(MessageHandler(Filters.photo, image_handler))


updater.start_polling()
updater.idle()
