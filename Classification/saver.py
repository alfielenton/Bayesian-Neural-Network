import json
import os
import torch

class Saver:

    def __init__(self):

        self.results_file = "results.json"
        self.target_file = "targets.txt"
        self.model_file = "model.pth"

        with open(self.results_file, "w") as f:
            json.dump({}, f)

    def save_targets(self, targets):
        with open(self.target_file, "w") as f:
            stt = ""
            for t in targets:
                stt += t + "\n"
            f.write(stt)

    def save_results(self, epoch, loss, metric):

        with open(self.results_file, "r") as f:
            data = json.load(f)

        if str(epoch) not in data.keys():
            data[str(epoch)] = {"loss":[], 
                           "metric":[]}

        data[str(epoch)]["loss"].append(loss)
        data[str(epoch)]["metric"].append(metric)

        with open(self.results_file, "w") as f:
            json.dump(data, f)

    def save_model(self, model_state_dict):
        torch.save(model_state_dict, self.model_file)