import torch
import torch.utils.data
import torch.cuda.amp
import torch.optim
import torch.optim.lr_scheduler
import torch.utils.tensorboard.writer as writer
import transformers

import os
import tqdm

import src.data, src.model

__all__ = ["Trainer"]


class Trainer:
    def __init__(
        self, device: str, max_epoch: int, accumu_steps: int, 
        ckpt_save_fold: str, ckpt_load_path: str, ckpt_load_lr: bool,
        batch_size: int, num_workers: int, lr: float, T_max: int, 
        trainset: src.data.DNADataset, model: src.model.DNABERT2FC,
    ) -> None:
        # train
        self.device = device
        self.max_epoch = max_epoch
        self.accumu_steps = accumu_steps
        # checkpoint
        self.ckpt_save_fold = ckpt_save_fold
        self.ckpt_load_path = ckpt_load_path
        self.ckpt_load_lr   = ckpt_load_lr

        # data
        self.trainloader = torch.utils.data.DataLoader(
            dataset=trainset, batch_size=batch_size, num_workers=num_workers, 
            persistent_workers=True, pin_memory=True, shuffle=True, 
        )
        # tokenizer, sequence [str] -> token [Tensor]
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "zhihan1996/DNABERT-2-117M", trust_remote_code=True
        )
        # model
        self.model = model.to(self.device)
        # optimizer
        self.scaler    = torch.cuda.amp.GradScaler()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=T_max
        )
        # recorder
        self.writer = writer.SummaryWriter()

        # index
        self.epoch = 1  # epoch index may update in load_ckpt()

        # print model info
        """
        para_num = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f'The model has {para_num:,} trainable parameters')
        """

    def fit(self) -> None:
        #self._loadCkpt()
        for self.epoch in tqdm.tqdm(
            range(self.epoch, self.max_epoch+1), 
            total=self.max_epoch, desc=self.ckpt_save_fold, smoothing=0.0,
            unit="epoch", initial=self.epoch, dynamic_ncols=True,
        ):
            self._trainEpoch()
            self._updateLr()
            #self._saveCkpt()

    def _trainEpoch(self) -> None:
        self.model.train()

        # record: progress bar
        pbar = tqdm.tqdm(
            total=int(len(self.trainloader)/self.accumu_steps), 
            desc='train_epoch', leave=False, unit="steps", smoothing=1.0, 
            dynamic_ncols=True,
        )
        # record: tensorboard
        train_loss = []

        for i, (sequence, coord, label) in enumerate(self.trainloader):
            # put frames and labels in GPU
            token = self.tokenizer(
                sequence, return_tensors = 'pt', padding=True
            )["input_ids"].squeeze(0).to(self.device)
            coord = coord.to(self.device)
            label = label.to(self.device)

            # forward and backward
            with torch.cuda.amp.autocast(dtype=torch.float16):
                predis = self.model(token, coord)
                loss_value  = torch.nn.functional.mse_loss(predis, label)
                loss_value /= self.accumu_steps
            self.scaler.scale(loss_value).backward()

            # record: tensorboard
            train_loss.append(loss_value.item() / len(predis))

            # update model parameters
            if (i+1) % self.accumu_steps != 0: continue
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # record: tensorboard
            self.writer.add_scalars(
                'scalars/loss', {'train': torch.sum(torch.as_tensor(train_loss))}, 
                (self.epoch - 1) * len(self.trainloader) / self.accumu_steps + 
                (i + 1) / self.accumu_steps
            )  # average loss of each frame
            train_loss = []
            # record: progress bar
            pbar.update()

    @torch.no_grad()
    def _updateLr(self) -> None:
        # update learning rate
        self.scheduler.step()

        # record: tensorboard
        self.writer.add_scalar(
            'scalars/lr', self.optimizer.param_groups[0]['lr'], 
            self.epoch * len(self.trainloader) / self.accumu_steps
        )

    @torch.no_grad()
    def _saveCkpt(self) -> None:
        # file path checking
        if not os.path.exists(self.ckpt_save_fold): 
            os.makedirs(self.ckpt_save_fold)

        torch.save({
            'epoch': self.epoch,  # epoch index start from 1
            'model': self.model.state_dict(),
            'scaler': self.scaler.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, "{}/{}.ckpt".format(self.ckpt_save_fold, self.epoch))

    @torch.no_grad()
    def _loadCkpt(self) -> None:
        if self.ckpt_load_path == "": return
        ckpt = torch.load("{}.ckpt".format(self.ckpt_load_path))
        
        self.epoch = ckpt['epoch']+1  # start train from next epoch index
        self.model.load_state_dict(ckpt['model'], strict=False)
        self.scaler.load_state_dict(ckpt['scaler'])
        
        if not self.ckpt_load_lr: return
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
