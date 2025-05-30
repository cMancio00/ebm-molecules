import torch

from utils.validity import check_validity
from data_modules.QM9DataModule import QM9DataModule

ROOT = "./datasets/qm9"
if __name__ == "__main__":
    dm = QM9DataModule(data_dir=ROOT, batch_size=2, num_samples=20)
    dm.prepare_data()
    dm.setup("fit")
    t = dm.train_dataloader()
    data,y = next(iter(t))

    adj = data.adj
    x = data.x
    print(x[0][0])
    x[0][0] = torch.tensor([0., 0., 0., 1.])
    x[0][1] = torch.tensor([0., 0., 0., 1.])
    x[0][2] = torch.tensor([0., 0., 0., 1.])
    atomic_num_list = [6,7,8,9,0]
    check_validity(adj,x, atomic_num_list)