import os
from copy import deepcopy
import tqdm
import torch
import torch.nn.functional as F
import colorful
import numpy as np
import networkx as nx
from tensorboardX import SummaryWriter
from .reservoir import reservoir
from components import Net
from utils import BetaMixture1D

class DiverseCont(torch.nn.Module):
    """ Train Continual Model self-supervisedly
        Freeze when required to eval and finetune supervisedly using Purified Buffer.
    """
    def __init__(self, config, writer: SummaryWriter):
        super().__init__()
        self.config = config
        self.device = config['device']
        self.writer = writer

        self.diverse_buffer = reservoir['diverse'](config, config['diverse_buffer_size'], config['diverse_buffer_q_poa'])
        self.delay_buffer = reservoir['delay'](config, config['delayed_buffer_size'], config['delayed_buffer_q_poa'])

        self.base_step = 0
        self.base_ft_step = 0

        self.base = self.get_init_base(config)

    def get_init_base(self, config):
        """get initialized base model"""
        base = Net[config['net']](config)
        optim_config = config['optimizer']
        lr_scheduler_config = deepcopy(config['lr_scheduler'])
        lr_scheduler_config['options'].update({'T_max': config['base_train_epochs']})

        base.setup_optimizer(optim_config)
        base.setup_lr_scheduler(lr_scheduler_config)
        return base

    def get_init_base_ft(self, config):
        """get initialized eval model"""
        base_ft = Net[config['net'] + '_ft'](config)
        optim_config = config['optimizer_ft']
        lr_scheduler_config = config['lr_scheduler_ft']

        base_ft.setup_optimizer(optim_config)
        base_ft.setup_lr_scheduler(lr_scheduler_config)
        return base_ft

    def learn(self, x, y, corrupt, idx, step=None):
        x, y = x.cuda(), y.cuda()
        for i in range(len(x)):
            self.delay_buffer.update(imgs=x[i: i + 1], cats=y[i: i + 1], corrupts=corrupt[i: i + 1], idxs=idx[i: i + 1])
            if self.delay_buffer.is_full():
                self.train_self_base()
                imgs, cats, corrupts, idxs = x, y, corrupt, idx
                # TODO: how to sample clean/diverse sampels?
                idxs = np.arange(len(self.delay_buffer))
                self.update_diverse_buffer_full_info(idxs, step)

    def update_diverse_buffer_full_info(self, idx, step):
        self.diverse_buffer.update(
            imgs = self.delay_buffer.get('imgs')[idx],
            cats = self.delay_buffer.get('cats')[idx],
            corrupts=self.delay_buffer.get('corrupts')[idx],
            idx=self.delay_buffer.get('idxs')[idx]
        )
        self.delay_buffer.reset()
        print(colorful.bold_yellow(self.diverse_buffer.state('corrupts')).styled_string)
        self.writer.add_scalar(
            'buffer_corrupts', torch.sum(self.diverse_buffer.get('corrupts')), step)

    def train_self_base(self):
        """
        Self Replay. train base model with samples from delay and purified buffer
        """
        bs = self.config['base_batch_size']
        # If diverse buffer is full, train using it also
        # db_bs: delay buffer batch size
        # divb_bs: diverse buffer batch size
        db_bs = (bs // 2) if self.diverse_buffer.is_full() else bs
        db_bs = min(db_bs, len(self.delay_buffer))
        divb_bs = min(bs - db_bs, len(self.diverse_buffer))

        self.base.train()
        self.base.init_ntxent(self.config, batch_size=db_bs + divb_bs)

        # Remember
        # ***_buffer.get_dataloader(self, batch_szie, drop_last, shuffle)
        #   -> return DataLoader
        # ***_buffer.sample(self, num, cat=None)
        #   -> return samples

        dataloader = self.delay_buffer.get_dataloader(batch_size=db_bs, shuffle=True, drop_last=True)

        for epoch_i in tqdm.trange(self.config['base_train_epochs'], desc="base training", leave=False):
            for inner_step, data in enumerate(dataloader):
                x = data['imgs']
                self.base.zero_grad()
                # sample data from diverse buffer and merge
                if divb_bs > 0:
                    replay_data = self.diverse_buffer.sample(num=divb_bs)
                    x = torch.cat([replay_data['imgs'], x], dim=0)

                loss = self.base.get_selfsup_loss(x)
                loss.backward()
                self.base.optimizer.step()

                self.writer.add_scalar(
                    'continual_base_train_loss', loss,
                    self.base_step + inner_step + epoch_i * len(dataloader))

            # warmup for the first 10 epochs
            if epoch_i >= 10:
                self.base.lr_scheduler.step()

        self.writer.flush()
        self.base_step += self.config['base_train_epochs'] * len(dataloader)

    def get_finetuned_model(self):
        """copy the base and fine-tune for evaluation"""
        base_ft = self.get_init_base_ft(self.config)
        # overwrite entries in the state dict
        ft_dict = base_ft.state_dict()
        ft_dict.update({k: v for k, v in self.base.state_dict().items() if k in ft_dict})
        base_ft.load_state_dict(ft_dict)

        base_ft.train()
        dataloader = self.diverse_buffer.get_dataloader(batch_size=self.config['ft_batch_size'], shuffle=True, drop_last=True)
        for epoch_i in tqdm.trange(self.config['ft_epochs'], desc='finetuning', leave=False):
            for inner_step, data in enumerate(dataloader):
                x, y = data['imgs'], data['cats']
                base_ft.zero_grad()
                loss = base_ft.get_sup_loss(x, y).mean()
                loss.backward()
                base_ft.clip_grad()
                base_ft.optimizer.step()
                base_ft.lr_scheduler.step()

                self.writer.add_scalar(
                    'ft_train_loss', loss,
                    self.base_ft_step + inner_step + epoch_i * len(dataloader))

        self.writer.flush()
        self.base_ft_step += self.config['ft_epochs'] * len(dataloader)
        base_ft.eval()
        return base_ft

    def forward(self, x):
        pass
