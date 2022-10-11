import os
import wandb
from os.path import join

import torch
from torch.nn.utils import clip_grad_norm_

from utils import cal_loss, cal_epsilon
from tqdm.auto import tqdm

class Trainer(object):
    def __init__(self, optimizer, model, lr_scheduler,
                 train_loader, val_loader, args,
                 use_cuda=True, init_epoch=1, last_epoch=15):

        self.optimizer = optimizer
        self.model = model
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args

        self.step = 0
        self.epoch = init_epoch
        self.total_step = (init_epoch-1)*len(train_loader)
        self.last_epoch = last_epoch
        self.best_val_loss = 1e18
        self.device = torch.device("cuda" if use_cuda else "cpu")
        
        wandb.init(project="Image to LaTex", entity="gr7-cviuh")
        wandb.config = {
            "learning_rate": self.args.lr,
            "min_learning_rate": self.args.min_lr,
            "epochs": self.args.epoches,
            "batch_size": self.args.batch_size,
            "embeding_size": self.args.emb_dim,
            "max_length_of_formula": self.args.max_len,
            "dropout_probility": self.args.dropout,
            "dec_rnn_h": self.args.dec_rnn_h,
            "save_dir": self.args.save_dir,
            "seed": self.args.seed,
            "print_freq": self.args.print_freq
        }

    def train(self):
        while self.epoch <= self.last_epoch:
            self.model.train()
            losses = 0.0
            dset = tqdm(iter(self.train_loader))
            for imgs, tgt4training, tgt4cal_loss in dset:
                step_loss = self.train_step(imgs, tgt4training, tgt4cal_loss)
                losses += step_loss

                # log message
                if self.step % self.args.print_freq == 0:
                    avg_loss = losses / self.args.print_freq
                    wandb.log({
                        "epoch": self.epoch,
                        "train_loss": avg_loss,
                        "train_perplexity": 2**avg_loss,
                    })
                    losses = 0.0
                dset.set_description("Epoch {}, train loss:{:.4f}".format(self.epoch, step_loss))
            
            # one epoch Finished, calcute val loss
            val_loss = self.validate()
            self.lr_scheduler.step(val_loss)

            self.save_model('ckpt-{}-{:.4f}'.format(self.epoch, val_loss))
            self.epoch += 1
            self.step = 0

    def train_step(self, imgs, tgt4training, tgt4cal_loss):
        self.optimizer.zero_grad()

        imgs = imgs.to(self.device)
        tgt4training = tgt4training.to(self.device)
        tgt4cal_loss = tgt4cal_loss.to(self.device)
        epsilon = cal_epsilon(
            self.args.decay_k, self.total_step, self.args.sample_method)
        logits = self.model(imgs, tgt4training, epsilon)

        # calculate loss
        loss = cal_loss(logits, tgt4cal_loss)
        self.step += 1
        self.total_step += 1
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.args.clip)
        self.optimizer.step()

        return loss.item()

    def validate(self):
        self.model.eval()
        val_total_loss = 0.0
        with torch.no_grad():
            dset = tqdm(iter(self.val_loader))
            for imgs, tgt4training, tgt4cal_loss in dset:
                imgs = imgs.to(self.device)
                tgt4training = tgt4training.to(self.device)
                tgt4cal_loss = tgt4cal_loss.to(self.device)

                epsilon = cal_epsilon(
                    self.args.decay_k, self.total_step, self.args.sample_method)
                logits = self.model(imgs, tgt4training, epsilon)
                loss = cal_loss(logits, tgt4cal_loss)
                val_total_loss += loss

                dset.set_description("Epoch {}, valid loss: {:.4f}".format(self.epoch, loss))
            avg_loss = val_total_loss / len(self.val_loader)
            wandb.log({
                "epoch": self.epoch,
                "val_loss": avg_loss,
                "val_perplexity": 2**avg_loss,
            })
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_model('best_ckpt')
        return avg_loss

    def save_model(self, model_name):
        if not os.path.isdir(self.args.save_dir):
            os.makedirs(self.args.save_dir)
        save_path = join(self.args.save_dir, model_name+'.pt')
        print("Saving checkpoint to {}".format(save_path))

        # torch.save(self.model, model_path)

        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_sche': self.lr_scheduler.state_dict(),
            'epoch': self.epoch,
            'args': self.args
        }, save_path)
