import os
import torch
import torchvision
from torchvision.transforms import v2

class Handler:

    def __init__(self):

        data_path = "datasets//"
        dts = ["train", "valid"]

        self.all_paths = []
        self.targets_to_paths = {}

        for dt in dts:
            dt_path = data_path + dt
            for folder in os.listdir(dt_path):
                if folder not in self.targets_to_paths.keys():
                    self.targets_to_paths[folder] = []
                bird_path = dt_path + "//" + folder
                for im in os.listdir(bird_path):
                    self.all_paths.append(bird_path + "//" + im)
                    self.targets_to_paths[folder].append(bird_path + "//" + im)

        self.targets = list(self.targets_to_paths.keys())
        self.num_classes = len(self.targets_to_paths.keys())
        self.num_data = len(self.all_paths)

    def gather_data(self, exclude: list | None):

        if exclude is None:
            return self.all_paths
        paths = []
        self.current_targets = []
        for bird in self.targets_to_paths:
            if bird not in exclude:
                self.current_targets.append(bird)
                paths += self.targets_to_paths[bird]
        
        return paths, self.current_targets

    def build_batch(self, paths):

        ims = []
        targets = []
        for p in paths:
            assert p in self.all_paths
            im = torchvision.io.read_image(p)
            ims.append(v2.Resize((200, 200))(im))
            for bird in self.targets_to_paths:
                if p in self.targets_to_paths[bird]:
                    targets.append(self.current_targets.index(bird))
        
        ims = torch.stack(ims) / 255.
        targets = torch.tensor(targets)

        return ims, targets