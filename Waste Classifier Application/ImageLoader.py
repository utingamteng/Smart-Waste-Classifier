import torch
from torchvision import transforms, models

ClassNames = [
               'battery', 
               'biological', 
               'cardboard', 
               'clothes', 
               'glass', 
               'metal', 
               'paper', 
               'plastic', 
               'shoes', 
               'trash'
               ]


class ImageClassifier:
    def __init__(self, ModelPath, device='cpu'):
        self.device = device

        self.model = models.convnext_tiny(weights=None)

        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = torch.nn.Linear(in_features, 10)

        state = torch.load(ModelPath, map_location=device)
        self.model.load_state_dict(state)

        self.model = self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, pil_image, ClassNames):
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)

        probs = torch.softmax(output, dim=1)
        conf, ClassIndex = torch.max(probs, dim=1)

        return (ClassNames[ClassIndex.item()], conf.item(), probs.squeeze().tolist())