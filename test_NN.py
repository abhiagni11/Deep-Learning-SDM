import numpy as np 
import torch
from artifact_prediction import Net

def predict_artifact(image):
    image = torch.Tensor(image)
    image = image.reshape(1, 1, image.shape[0],image.shape[1])
    
    model = Net()
    model.load_state_dict(torch.load('mytraining.pth'))    
    outputs = model(image)
    return outputs

images = np.random.randint(2,size=(16,16))
print(predict_artifact(images).shape)