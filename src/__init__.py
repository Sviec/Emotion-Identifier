from src.model import train, train_fine
from src.scripts import test
from config import EFFICIENTNET

if __name__ == '__main__':
    train(EFFICIENTNET)
    train_fine(EFFICIENTNET)
    test(EFFICIENTNET)
