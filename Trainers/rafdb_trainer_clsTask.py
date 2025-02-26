import numpy as np
import pandas as pd
import sys
import datetime
import os
import traceback
import shutil

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# from utils.radam import RAdam

import tqdm
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiplicativeLR, StepLR, MultiStepLR, ConstantLR, LinearLR, PolynomialLR, CosineAnnealingLR, ChainedScheduler, ExponentialLR, SequentialLR, ReduceLROnPlateau, CyclicLR, CosineAnnealingWarmRestarts

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../Utils/Metrics')))
from classify_metrics import accuracy, make_batch

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def create_CSV(output_csv_path):
  df = pd.DataFrame(columns=['epoch', 'learning_rate','accuracy', 'loss', 'val_accuracy', 'val_loss'])
  df.to_csv(output_csv_path, index=False)

def update_output_csv(output_csv_path, epoch, lr, accuracy, loss, val_accuracy, val_loss):
  df = pd.read_csv(output_csv_path)
  new_line = pd.DataFrame({'epoch':[epoch], 'learning_rate':[lr],'accuracy':[accuracy], 'loss':[loss], 'val_accuracy':[val_accuracy], 'val_loss':[val_loss]})
  new_line = new_line[df.columns]
  df = pd.concat([df, new_line], ignore_index=True)
  df.to_csv(output_csv_path, index=False)

class Trainer(object):
    """base class for trainers"""

    def __init__(self):
        pass


class RAFDB_Trainer(Trainer):
  def __init__(self, model, train_loader, val_loader, test_loader,test_loader_ttau, configs, wb = False, output_csv_path = '/kaggle/working/out.csv', initial_best_val_acc = 0.0):

    self.train_loader = train_loader
    self.val_loader = val_loader
    self.test_loader = test_loader
    self.test_loader_ttau = test_loader_ttau
    self.output_csv_path = output_csv_path
    create_CSV(output_csv_path = self.output_csv_path)
    self.configs = configs

    self.batch_size = configs["batch_size"]
    # self.epochs = configs["epochs"]
    self.learning_rate = configs["lr"]
    self.min_lr = configs["min_lr"]
    self.num_workers = configs["num_workers"]
    self.momentum = configs["momentum"]
    self.weight_decay = configs["weight_decay"]
    self.device = torch.device(configs["device"])
    self.max_plateau_count = configs["max_plateau_count"]
    self.max_epoch_num = configs["max_epoch_num"]
    self.distributed = configs["distributed"]
    self.optimizer_chose = configs["optimizer_chose"]
    self.lr_scheduler_chose = configs["lr_scheduler"]
    self.name_run_wandb = configs["name_run_wandb"]
    self.wb = wb

    #self.model = model.to(self.device)'cpu'
    '''if torch.cuda.is_available():
      self.device = torch.device('cuda:0')  # Use CUDA device 
    else:
      self.device = torch.device('cpu')
      print("CUDA is not available, falling back to CPU.")'''
    self.model = model.to(self.device)

# Move the model to the device
    '''try:
      model = model.to(self.device)
    except Exception as e:
      print("Error:", e)'''

    self.train_loss_list = []
    self.train_acc_list = []
    self.val_loss_list = []
    self.val_acc_list = []
    self.best_train_acc = 0.0
    self.best_val_acc = initial_best_val_acc
    self.best_train_loss = 0.0
    self.best_val_loss = 0.0
    self.test_acc = 0.0
    self.test_acc_ttau = 0.0
    self.plateau_count = 0
    #self.current_epoch_num = 0
    self.current_epoch_num = configs["current_epoch_num"]
    # Set information for training
    self.start_time = datetime.datetime.now()

    self.checkpoint_dir = "/kaggle/working/"

    '''self.checkpoint_path = os.path.join(self.checkpoint_dir, "{}_{}_{}".format
                                        (self.configs["project_name"], self.configs["model"], self.start_time.strftime("%Y%b%d_%H.%M"),))'''

    self.checkpoint_path = os.path.join(self.checkpoint_dir,"ResnetDuck_Cbam_cuaTuan")

    if self.distributed == 1:
            torch.distributed.init_process_group(backend="nccl")
            self.model = nn.parallel.DistributedDataParallel(self.model)
            print("Let's use", torch.cuda.device_count(), "GPUs!")

            self.train_ds = DataLoader(
                self.train_loader,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=True,
                worker_init_fn=lambda x: np.random.seed(x),
            )
            self.val_ds = DataLoader(
                self.val_loader,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=False,
                worker_init_fn=lambda x: np.random.seed(x),
            )

            self.test_ds = DataLoader(
                self.test_loader,
                batch_size=1,
                num_workers=self.num_workers,
                pin_memory=True,
                shuffle=False,
                worker_init_fn=lambda x: np.random.seed(x),
            )
    else:

      self.train_ds = DataLoader(self.train_loader,batch_size=self.batch_size,num_workers=self.num_workers,
                        pin_memory=True, shuffle=True)
      self.val_ds = DataLoader(self.val_loader, batch_size = self.batch_size, num_workers=self.num_workers,
                      pin_memory=True, shuffle=False)
      self.test_ds = DataLoader(self.test_loader, batch_size= 1,num_workers=self.num_workers,
                      pin_memory=True, shuffle=False)
    
    
    self.criterion = nn.CrossEntropyLoss().to(self.device)

    if self.optimizer_chose == "RAdam":
      print("The selected optimizer is RAdam")
      self.optimizer = torch.optim.RAdam(
        params=self.model.parameters(),
        lr=self.learning_rate,
        weight_decay=self.weight_decay,
#       amsgrad = True,
    )
    elif self.optimizer_chose == "SGD":
      print("The selected optimizer is SGD")
      self.optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
    elif self.optimizer_chose == "RMSprop":
      print("The selected optimizer is RMSprop")
      self.optimizer = torch.optim.RMSprop(
        params=model.parameters(),
        lr=self.learning_rate,
        weight_decay=self.weight_decay,
        momentum=0.9,
        alpha=0.99,
        eps=1e-8)
    elif self.optimizer_chose == "Adam":
      print("The selected optimizer is Adam")
      self.optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=self.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=self.weight_decay)
    elif self.optimizer_chose == "AdamW":
      print("The selected optimizer is AdamW")
      self.optimizer = torch.optim.AdamW(
          params=self.model.parameters(),
          lr=self.learning_rate,
          betas=(0.9, 0.999),
          eps=1e-8,
          weight_decay=self.weight_decay)
    elif self.optimizer_chose == "Adamax":
      print("The selected optimizer is Adamax")
      self.optimizer = torch.optim.Adamax(
          params=self.model.parameters(),
          lr=self.learning_rate,
          betas=(0.9, 0.999),
          eps=1e-8,
          weight_decay=self.weight_decay)
    elif self.optimizer_chose == "Adagrad":
      print("The selected optimizer is Adagrad")
      self.optimizer = torch.optim.Adagrad(
          params=self.model.parameters(),
          lr=self.learning_rate,
          lr_decay=0.001,  
          weight_decay=self.weight_decay,  
          initial_accumulator_value=0.1,  
          eps=1e-8
      )
    else:  #default ="RAdam"
      print("The selected optimizer is RAdam")
      self.optimizer = torch.optim.RAdam(
        params=self.model.parameters(),
        lr=self.learning_rate,
        weight_decay=self.weight_decay,
#       amsgrad = True,
    )
      
      
    if self.lr_scheduler_chose == "ReduceLROnPlateau":
      self.scheduler = ReduceLROnPlateau(
        self.optimizer,
        patience=self.configs["plateau_patience"],
        min_lr=self.min_lr,
        # factor = torch.exp(torch.Tensor([-0.1])),
        factor = 0.1,
      )
      print("The selected learning_rate scheduler strategy is ReduceLROnPlateau")
    elif self.lr_scheduler_chose == "MultiStepLR":
      milestones = [x for x in range(5, 120, 5)]
      self.scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=0.5,  verbose=True)
      print("The selected learning_rate scheduler strategy is MultiStepLR")

    elif self.lr_scheduler_chose == "ExponentialLR":
      self.scheduler = ExponentialLR(self.optimizer, gamma = 0.8,  verbose=True)
      print("The selected learning_rate scheduler strategy is ExponentialLR")

    elif self.lr_scheduler_chose == "PolynomialLR":
      self.scheduler = PolynomialLR(self.optimizer, total_iters=30, power=2,  verbose=True)
      print("The selected learning_rate scheduler strategy is PolynomialLR")

    elif self.lr_scheduler_chose == "CosineAnnealingLR":
      self.scheduler = CosineAnnealingLR(self.optimizer, T_max=10, eta_min=self.min_lr,  verbose=True)
      print("The selected learning_rate scheduler strategy is CosineAnnealingLR")

    elif self.lr_scheduler_chose == "CosineAnnealingWarmRestarts":
      self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2, eta_min=self.min_lr,  verbose=True)
      print("The selected learning_rate scheduler strategy is CosineAnnealingWarmRestarts")

    else: #default ="ReduceLROnPlateau"
      self.lr_scheduler_chose = 'None'
      lambda_lr = lambda epoch: 1.0  # Không thay đổi learning rate
      self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda_lr)
      print(f"No choosing Learning rate scheduler(lr={self.learning_rate})")

       
  def init_wandb(self):
    #set up wandb for training
    if self.wb == True:
      try:
        print("------------SETTING UP WANDB--------------")
        import wandb
        self.wandb = wandb
        self.wandb.login(key=self.configs["wandb_api_key"])
        print("------Wandb Init-------")

        self.wandb.init(
            project = self.configs["project_name"],
            name = self.name_run_wandb,
            config = self.configs
        )
        self.wandb.watch(self.model, self.criterion, log="all", log_freq=10)
        print()
        print("-----------------------TRAINING MODEL-----------------------")
      except:
          print("--------Can not import wandb-------")

    # return wandb
  def step_per_train(self):
    # if self.wb == True:
    #   self.wandb.watch(model)

    self.model.train()
    train_loss = 0.0
    train_acc = 0.0

    for i, (images, labels) in tqdm.tqdm(
        enumerate(self.train_ds), total = len(self.train_ds), leave = True, colour = "blue", desc = f"Epoch {self.current_epoch_num}",
        bar_format="{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    ):

      # Move images to GPU before feeding them to the model, to fix error happen : Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
      self.model = self.model.cuda()
      
      images = images.cuda(non_blocking = True)
      labels = labels.cuda(non_blocking = True)

      # compute output, accuracy and get loss
      y_pred = self.model(images)
    
      loss = self.criterion(y_pred, labels)
      acc = accuracy(y_pred, labels)[0]
      

      train_loss += loss.item()
      train_acc += acc.item()

      # compute gradient and do SGD step
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      # write wandb
      metric = {
          " Loss" : train_loss / (i+1),
          " Accuracy" :train_acc / (i+1),
          " epochs" : self.current_epoch_num,
          " Learning_rate" : get_lr(self.optimizer)
      }
      if self.wb == True and i <= len(self.train_ds):
            self.wandb.log(metric)

      
    i += 1
    self.train_loss_list.append(train_loss / i)
    self.train_acc_list.append(train_acc / i)


    print(" Loss: {:.4f}".format(self.train_loss_list[-1]), ", Accuracy: {:.2f}%".format(self.train_acc_list[-1]))

  def step_per_val(self):
    self.model.eval()
    val_loss = 0.0
    val_acc = 0.0

    with torch.no_grad():
      for i, (images, labels) in tqdm.tqdm(
          enumerate(self.val_ds), total = len(self.val_ds), leave = True, colour = "green", desc = "        ",
          bar_format="{desc} {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
      ):
        images = images.cuda(non_blocking = True)
        labels = labels.cuda(non_blocking = True)

        # compute output, accuracy and get loss
        y_pred = self.model(images)

        loss = self.criterion(y_pred, labels)
        acc = accuracy(y_pred, labels)[0]

        val_loss += loss.item()
        val_acc += acc.item()

      i += 1
      self.val_loss_list.append(val_loss / i)
      self.val_acc_list.append(val_acc / i)

      print(" Val_Loss: {:.4f}".format(self.val_loss_list[-1]),", Val_Accuracy: {:.2f}%".format(self.val_acc_list[-1]),", Learning_rate: {:.7}".format(self.optimizer.param_groups[0]['lr']))

      # write wandb
      if self.wb == True:
        metric = {
            " Val_Loss" : self.val_loss_list[-1],
            " Val_Accuracy" :self.val_acc_list[-1],
            # "Learning_rate" : self.learning_rate
        }
        self.wandb.log(metric)

      

  def acc_on_test(self):
    self.model.eval()
    test_loss = 0.0
    test_acc = 0.0

    with torch.no_grad():
      for i, (images, labels) in tqdm.tqdm(
          enumerate(self.test_ds), total = len(self.test_ds), leave = True, colour = "green", desc = "        ",
          bar_format="{desc} {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
      ):
        images = images.cuda(non_blocking = True)
        labels = labels.cuda(non_blocking = True)

        # compute output, accuracy and get loss
        y_pred = self.model(images)

        loss = self.criterion(y_pred, labels)
        acc = accuracy(y_pred, labels)[0]

        test_loss += loss.item()
        test_acc += acc.item()

        # print(i)
      i += 1
      test_loss = (test_loss / i)
      test_acc = (test_acc / i)

      print("Accuracy on Test_ds: {:.3f}".format(test_acc))
      if self.wb == True:
        self.wandb.log({"Test_accuracy": test_acc})
      return test_acc

  def acc_on_test_ttau(self):
    self.model.eval()
    test_acc = 0.0
    # print(" Calculate accuracy on Test_ds with TTAU...!")

    # write log for testting

    with torch.no_grad():
      for idx in tqdm.tqdm(
          range(len(self.test_loader_ttau)), total = len(self.test_loader_ttau), leave = False
      ):

        images, labels = self.test_loader_ttau[idx]
        labels = torch.LongTensor([labels])

        images = make_batch(images)
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)

        y_pred = self.model(images)
        y_pred = F.softmax(y_pred, 1)

        y_pred = torch.sum(y_pred, 0)

        y_pred = torch.unsqueeze(y_pred, 0)

        acc = accuracy(y_pred, labels)[0]

        test_acc += acc.item()

      test_acc = test_acc / (idx + 1)
    print("Accuracy on Test_ds with TTAU: {:.3f}".format(test_acc))
    if self.wb == True:
      self.wandb.log({"Testta_accuracy": test_acc})

    return test_acc

  def Train_model(self):
    self.init_wandb()
    #self.scheduler.step(100 - self.best_val_acc)
    try:
      while not self.stop_train():
        self.update_epoch_num()
        self.step_per_train()
        self.step_per_val()

        self.update_state_training()

    except KeyboardInterrupt:
      traceback.print_exc()
      pass
    # Stop training
    try:
      #loading best model
      state = torch.load(self.checkpoint_path)
      self.model.load_state_dict(state["net"])
      print("----------------------Cal on Test-----------------------")
      self.test_acc = self.acc_on_test()
      self.test_acc_ttau = self.acc_on_test_ttau()
      self.save_weights()

    except Exception as e:
      traceback.prtin_exc()
      pass

    consume_time = str(datetime.datetime.now() - self.start_time)
    print("----------------------SUMMARY-----------------------")
    print(" After {} epochs and {} plateau count, consume {}".format((self.current_epoch_num), (self.plateau_count),consume_time[:-7]))
    print(" Best Accuracy on Train: {:.3f} ".format(self.best_train_acc))
    print(" Best Accuracy on Val: {:.3f} ".format(self.best_val_acc))
    print(" Best Accuracy on Test: {:.3f} ".format(self.test_acc))
    print(" Best Accuracy on Test with tta: {:.3f} ".format(self.test_acc_ttau))
    return self.model, self.best_val_acc

  #set up for training (update epoch, stopping training, write logging)
  def update_epoch_num(self):
    self.current_epoch_num += 1

  def stop_train(self):
    return (
        self.plateau_count > self.max_plateau_count or
        self.current_epoch_num > self.max_epoch_num
    )
  
  def update_state_training(self):
    if self.val_acc_list[-1] > self.best_val_acc:
      self.save_weights()
      self.plateau_count = 0
      self.best_val_acc = self.val_acc_list[-1]
      self.best_val_loss = self.val_loss_list[-1]
      self.best_train_acc = self.train_acc_list[-1]
      self.best_train_loss = self.train_loss_list[-1]
      print(f'Weight was updated because val_accuracy get highest(={self.val_acc_list[-1]})')
    else:
      self.plateau_count += 1
    
    #update CSV
    update_output_csv(output_csv_path = self.output_csv_path, 
                      epoch=len(self.val_acc_list),
                      lr = self.optimizer.param_groups[0]['lr'], 
                      accuracy = self.train_acc_list[-1],
                      loss = self.train_loss_list[-1],
                      val_accuracy = self.val_acc_list[-1],
                      val_loss = self.val_loss_list[-1])

# 100 - self.best_val_acc
    if self.lr_scheduler_chose == "ReduceLROnPlateau":
      self.scheduler.step(self.val_acc_list[-1])
    else:
      self.scheduler.step()

    if self.optimizer.param_groups[0]['lr'] < self.min_lr:
      self.optimizer.param_groups[0]['lr'] = self.min_lr
#100 - self.best_val_acc

  def save_weights(self):
    state_dict = self.model.state_dict()

    state = {
        **self.configs,
        "net": state_dict,
        "best_val_loss": self.best_val_loss,
        "best_val_acc": self.best_val_acc,
        "best_train_loss": self.best_train_loss,
        "best_train_acc": self.best_train_acc,
        "train_loss_list": self.train_loss_list,
        "val_loss_list": self.val_loss_list,
        "train_acc_list": self.train_acc_list,
        "val_acc_list": self.val_acc_list,
        "test_acc": self.test_acc,
        "optimizer": self.optimizer.state_dict(),
    }

    torch.save(state, self.checkpoint_path)