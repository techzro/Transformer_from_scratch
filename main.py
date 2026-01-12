from training.train import train_model
from config import get_config

if __name__ == "__main__":
    config = get_config()
    train_model(config)
