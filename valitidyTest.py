from utils.validity import check_validity
from data_modules.QM9DataModule import QM9DataModule

ROOT = "./datasets/qm9"
if __name__ == "__main__":
    print("Loading data...")
    dm = QM9DataModule(data_dir=ROOT, batch_size=2, num_samples=20)
    print("Preparing data...")
    dm.prepare_data()
    dm.setup("fit")
    print("Preparing data loader...")
    t = dm.train_dataloader()
    data,y = next(iter(t))
    print("Done...")

    adj = data.adj
    x = data.x
    atomic_num_list = [6,7,8,9,0]
    check_validity(adj,x, atomic_num_list)