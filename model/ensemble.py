import torch
import torch.nn as nn
from model.CNN import CNN
import torch.nn.functional as F
from model.CNN_residual import CNN_Residual

class Ensemble(nn.Module):
    def __init__(self, model, model2, model3, num_classes=3):
        super(Ensemble, self).__init__()
        self.origin_model = model
        self.canny_model = model2
        self.sobel_model = model3
        self.origin_model.classifier = nn.Identity()
        self.canny_model.classifier = nn.Identity()
        self.sobel_model.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
			nn.Linear(64*53*100*2 + 64*213*100, 1024),
			nn.ReLU(),
			nn.Dropout(p=0.2),
            nn.Linear(1024, 512),
			nn.ReLU(),
			nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
		)

    def forward(self, x):
        origin = self.origin_model(x.clone())
        origin = origin.view(origin.size(0), -1)
        canny = self.canny_model(x.clone())
        canny = canny.view(canny.size(0), -1)
        sobel = self.sobel_model(x)
        sobel = sobel.view(sobel.size(0), -1)
        out = torch.cat((origin, canny, sobel), dim=1)
        out = self.classifier(F.relu(out))
        return out

def EnsembleNetwork(weight_path, weight_path2, weight_path3):
    modelA = CNN()
    modelA.load_state_dict(torch.load(weight_path))
    modelB = CNN_Residual('canny')
    modelB.load_state_dict(torch.load(weight_path2))
    modelC = CNN_Residual('sobel')
    modelC.load_state_dict(torch.load(weight_path3))
    return Ensemble(model=modelA, model2=modelB, model3=modelC)