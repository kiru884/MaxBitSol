import os
import yaml
import torch
import json
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
from pprint import pprint

from ..model import MODELS
from ..optimizers import OPTIMIZERS
from ..processing import DATASETS
from ..scores import SCORES


class BaseTrainer():
    def __init__(self, config, run=None):
        # load configs
        self.data_config = config["Data"]
        self.params_config = config["Parameters"]
        self.model_config = config["Model"]

        # neptune writer
        self.run = run

        # create results folder with model weights
        rp = os.path.join(self.data_config["result_root_path"], self.params_config["exp_name"])
        self.results_path = {
            "root": rp,
            "model": os.path.join(rp, "model"),
            "info": os.path.join(rp, "info")
        }
        if os.path.exists(rp):
            raise FileExistsError(f"There is already exists results file {rp}")
        else:
            os.mkdir(self.results_path["root"])
            os.mkdir(self.results_path["model"])
            os.mkdir(self.results_path["info"])
            # save config file
            with open(os.path.join(rp, "config.yaml"), 'w') as yamlfile:
                yaml.dump(config, yamlfile, default_flow_style=False)

        # load model
        self.device = self.model_config["device"]
        self.model = MODELS[self.model_config["name"]](self.model_config, self.device)

        # load data iterators
        training_set = DATASETS[self.data_config["dataset_type"]](self.data_config, train=True)
        validation_set = DATASETS[self.data_config["dataset_type"]](self.data_config, train=False)
        self.training_loader = DataLoader(training_set,
                                          batch_size=self.params_config["batch_size"],
                                          num_workers=self.params_config["num_workers"],
                                          pin_memory=True,
                                          pin_memory_device=self.device,
                                          shuffle=True)
        self.validation_loader = DataLoader(validation_set,
                                            batch_size=self.params_config["batch_size"],
                                            num_workers=self.params_config["num_workers"],
                                            pin_memory=True,
                                            pin_memory_device=self.device,
                                            shuffle=False)
        # totals
        self.train_total = (len(training_set) // self.params_config["batch_size"]) + 1
        self.val_total = (len(validation_set) // self.params_config["batch_size"]) + 1
        self.ep_total = self.params_config["num_epochs"]

        # load losses and metrics
        self.loss = SCORES[self.params_config["loss"]['name']](**self.params_config["loss"]['params'])
        self.metrics = dict()
        for m in self.params_config["metrics"]:
            m_name = m["params"].pop("name")
            self.metrics[m["title"]] = SCORES[m_name](m["params"]) if m["params"] else SCORES[m_name]()

        # load optimizer
        opt = OPTIMIZERS[self.params_config["opt"]["name"]]
        opt_params = {'params': self.model.parameters()}
        opt_params.update(self.params_config["opt"]["params"])
        self.optimizer = opt(**opt_params)

        # load scheduler
        self.scheduler = None
        if self.params_config["scheduler"]["name"] != 'None':
            scheduler = OPTIMIZERS[self.params_config["scheduler"]["name"]]
            self.scheduler = scheduler(self.optimizer, **self.params_config["scheduler"]["params"])


    def _run_train_epoch(self):
        sum_loss = 0
        all_true_labels = []
        all_pred_scores = []
        for i, data in tqdm(enumerate(self.training_loader), desc="Train loop...", total=self.train_total,
                            position=0, leave=True):
            # load data
            samples, labels = data["sample"], data["label"]
            samples = [s.to(torch.device(self.device)) for s in samples]
            samples = samples[0] if len(samples) == 1 else samples
            labels = labels.to(torch.device(self.device))
            # if i == 30:
            #     break

            # zero grads and get predicts
            self.optimizer.zero_grad()
            pred_scores = self.model(samples)

            # print("\n\n\n")
            # print(pred_scores)
            # print(labels)

            # get loss, backward
            loss = self.loss(labels, pred_scores)
            loss.backward()
            self.optimizer.step()

            # get losses and metrics
            sum_loss += loss

            # collect pred / true labels
            all_true_labels.append(labels.cpu().detach())
            all_pred_scores.append(pred_scores.cpu().detach())

        # collect all labels and predicts
        all_true_labels = torch.concat(all_true_labels, dim=0).numpy()
        all_pred_scores = torch.concat(all_pred_scores, dim=0).numpy()

        # print("INFO")
        # print(all_pred_scores)
        # print(all_true_labels)

        # get metrics
        info = {"metrics": {"train/loss": sum_loss.item() / (i + 1)}}
        for m, mf in self.metrics.items():
                info["metrics"][f"train/{m}"] = mf(all_true_labels, all_pred_scores)

        return info


    def _run_validation(self):
        sum_loss = 0
        all_true_labels = []
        all_pred_scores = []
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.validation_loader), desc="Validation loop...", total=self.val_total,
                                position=0, leave=True):
                # load data
                samples, labels = data["sample"], data["label"]
                samples = [s.to(torch.device(self.device)) for s in samples]
                samples = samples[0] if len(samples) == 1 else samples
                labels = labels.to(torch.device(self.device))

                # if i == 20:
                #     break

                # get predicts
                pred_scores = self.model(samples)
                sum_loss += self.loss(labels, pred_scores)

                # collect pred / true labels
                all_true_labels.append(labels.cpu().detach())
                all_pred_scores.append(pred_scores.cpu().detach())

        # collect all labels and predicts
        all_true_labels = torch.concat(all_true_labels, dim=0).numpy()
        all_pred_scores = torch.concat(all_pred_scores, dim=0).numpy()

        # get metrics
        info = {"metrics": {"val/loss": sum_loss.item() / (i + 1)}}
        info["true_labels"] = all_true_labels.tolist()
        info["pred_probs"] = all_pred_scores.tolist()
        for m, mf in self.metrics.items():
                info["metrics"][f"val/{m}"] = mf(all_true_labels, all_pred_scores)

        return info


    def train(self):
        for epoch in tqdm(range(self.params_config["num_epochs"]), desc="Training...", total=self.ep_total,
                          position=0, leave=True):
            self.model.train()
            train_info = self._run_train_epoch()

            self.model.eval()
            valid_info = self._run_validation()

            # scheduler procedures
            if self.scheduler is not None:
                lr_before_step = self.optimizer.param_groups[0]["lr"]
                if self.params_config["scheduler"]["name"] == "sch.ReduceLROnPlateau":
                    self.scheduler.step(valid_info["metrics"]["val/loss"])
                else:
                    self.scheduler.step()
                lr_after_step = self.optimizer.param_groups[0]["lr"]
                print(f"LR: {lr_before_step} -> {lr_after_step}")

            # report metrics
            self.report_metrics(train_info["metrics"], epoch)
            self.report_metrics(valid_info["metrics"], epoch)
            pprint(valid_info["metrics"])
            pprint(train_info["metrics"])

            # save results info
            info_save_path = os.path.join(self.results_path["info"], f"info_{epoch}.json")
            with open(info_save_path, "w") as fp:
                json.dump({"train": train_info, "valid": valid_info}, fp)
                fp.close()
            # save model
            model_save_path = os.path.join(self.results_path["model"], f"model_{epoch}.pt")
            torch.save(self.model.state_dict(), model_save_path)


    def report_metrics(self, metrics, epoch):
        if self.run is not None:
            for m_title, m_score in metrics.items():
                print((m_title, m_score))
                if type(m_score) in [np.array, list]:
                    for i, el_m_score in enumerate(m_score):
                        self.run[f"{m_title}_cls{i}"].append(float(el_m_score))
                else:
                    self.run[m_title].append(float(m_score))


