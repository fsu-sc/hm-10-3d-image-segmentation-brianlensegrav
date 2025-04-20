from mymodel import UNet3D
from training import train_model
from dataloader import get_dataloaders  # Implement separately

def main():
    dataloaders = get_dataloaders()  # Include transforms if needed
    model = UNet3D()
    train_model(model, dataloaders)

if __name__ == "__main__":
    main()
