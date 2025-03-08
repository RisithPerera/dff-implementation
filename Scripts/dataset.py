import os

from PIL import Image
from torch.utils.data import Dataset


class Market1501Dataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        pids = set()
        for img_name in os.listdir(root):
            if img_name.endswith('.jpg'):
                pid = int(img_name.split('_')[0])
                pids.add(pid)
                img_path = os.path.join(root, img_name)
                self.samples.append((img_path, pid))
        # Map original person IDs to contiguous labels
        self.pid2label = {pid: i for i, pid in enumerate(sorted(pids))}
        # Update samples with remapped labels
        self.samples = [(img_path, self.pid2label[pid]) for img_path, pid in self.samples]
        print(f'Found {len(self.samples)} samples from {len(pids)} identities.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, pid = self.samples[index]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, pid
