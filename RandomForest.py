from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import torch
from FragmentWithInkCropDataset import FragmentWithInkCropDataset
from torch.utils.data import DataLoader
from torchvision import transforms

class RFPixelClassifier():
    def __init__(self, input_image_size) -> None:
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        return self.model.score(X, y)
    


if __name__ == "__main__":
    model = RFPixelClassifier((100000//2, 100000//2))
    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = FragmentWithInkCropDataset('train/1', crop_size=1,transform=transforms)
    dataloader = DataLoader(dataset, batch_size=400, shuffle=True)
    for i, batch in enumerate(dataloader):
        images_batch, mask, target = batch
        print(f"images_batch.shape: {images_batch.shape}")
        print(f"mask.shape: {mask.shape}")
        print(f"target.shape: {target.shape}")
        if torch.sum(mask) == 0:
            print(f"Skipping batch {i} because sum(mask) == 0")
            continue
        if torch.sum(target) == 0:
            print(f"Skipping batch {i} because sum(target) == 0")
            continue
        # reshape the images_batch to be [35, 28*28] (35 is the batch size)
        images_batch = images_batch.reshape(-1,images_batch.shape[1])
        target = target.reshape(-1)
        print(f"images_batch.shape: {images_batch.shape}")
        X = images_batch
        y = target
        model.train(X, y)
        print("Training complete")
        print(f"{model.score(X, y)}")
        print("Predicting...")
        pred = model.predict_proba(X)
        print(f"pred.shape: {pred.shape}")
        print(f"pred: {pred[:,1]}")
        print(f"y: {y}")
        print(f"pred == y: {pred == y}")
        print(model.model.feature_importances_)
        plt.subplot(1, 2, 1)
        plt.imshow(images_batch.reshape(65, 20, 20)[0])
        plt.subplot(1, 2, 2)
        plt.imshow(pred[:,1].reshape(1,20, 20))
        plt.show()
        break


