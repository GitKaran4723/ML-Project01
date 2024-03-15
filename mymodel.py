import warnings
from fastai.vision.all import *
import timm
import os

def predict_image(image_path, num_classes=3):
    # Suppress warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    print("image_path: ", image_path)

    # Load the DataLoaders
    path = "./chest_xray/"
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock), 
        get_items=get_image_files, 
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(460),
        batch_tfms=[*aug_transforms(size=224), Normalize.from_stats(*imagenet_stats)]
    )
    
    dls = dblock.dataloaders(path, bs=64, types=(PILImage, TensorCategory))

    # Define the custom Xception architecture
    class XceptionModel(nn.Module):
        def __init__(self, num_classes, pretrained=True):
            super(XceptionModel, self).__init__()
            self.base_model = timm.create_model("xception", pretrained=pretrained)
            # Modify the last fully connected layer to match the number of classes
            self.fc = nn.Linear(self.base_model.num_features, num_classes)

        def forward(self, x):
            x = self.base_model.forward_features(x)
            x = F.adaptive_avg_pool2d(x, (1, 1)).reshape(x.size(0), -1)
            x = self.fc(x)
            return x

    # Create the Learner
    learn = Learner(dls, XceptionModel(num_classes=num_classes), metrics=accuracy)

    model_path = "./mymodel"
    learn.load(model_path)

    # Load the image
    img = PILImage.create(image_path)

    # Make a prediction
    prediction, _, details = learn.predict(img)

    # Extract accuracy from details
    prediction_accuracy = details[1].item()

    return prediction, prediction_accuracy
