import pandas as pd
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer, AdamW, get_cosine_schedule_with_warmup

from utils import set_seed

class MMRegressor(nn.Module):

    def __init__(self, model_path):
        
        super(MMRegressor, self).__init__()
        self.config = XLMRobertaConfig.from_pretrained(model_path)
        self.reg_model = XLMRobertaModel.from_pretrained(model_path)

        self.fc1 = nn.Linear(self.config.hidden_size, 512)
        self.fc2 = nn.Linear(512, 7)
        self.activation = nn.GELU()

    def forward(self, input_ids, attention_mask):

        output1 = self.reg_model(input_ids, attention_mask)[1]
        logits1 = self.fc2(self.activation(self.fc1(output1)))

        output2 = self.reg_model(input_ids, attention_mask)[1]
        logits2 = self.fc2(self.activation(self.fc1(output2)))
        
        return logits1, logits2

class Reg_FT_Configer():

    def __init__(self, params_dict: dict):
        
        super().__init__()
        
        self.learning_rate = params_dict['learning_rate']
        self.epoch =params_dict['epoch']
        self.gradient_acc = params_dict['gradient_acc']
        self.batch_size = params_dict['batch_size']
        self.max_len = params_dict['max_len']
        self.model_save_path = params_dict['model_save_path']
        self.warmup_rate = params_dict['warmup_rate']
        self.weight_decay = params_dict['weight_decay']
        self.model_pretrain_dir = params_dict['model_pretrain_dir']
        self.training_set_path = params_dict['training_set_path']
        self.testing_set_path = params_dict['testing_set_path']
        self.seed = params_dict['seed']

        # weights for the 7 sub-dimensions
        self.dims_weights = [params_dict['overall_weight'] if i == 4 else (1-params_dict['overall_weight'])/6 for i in range(7)]
        
        # weights for forward loss and adapted R-Drop loss
        self.losses_weights = {
            'forward_weight': (1-params_dict['rdrop_weight'])/2,
            'rdrop_weight': params_dict['rdrop_weight']
        }
        
        
class Reg_Trainer():

    def __init__(self, config: Reg_FT_Configer):

        super().__init__()

        self.config = config
        self.device = torch.device("cuda")
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.config.model_pretrain_dir)

        set_seed(self.config.seed)


    def dataset(self, data_path):

        input_ids, attention_masks, labels = [], [], []

        for idx, row in pd.read_csv(data_path).iterrows():
            text1, text2 = row['text1'], row['text2']
            encode_dict = self.tokenizer.__call__(text1,text2,
                                                    max_length=self.config.max_len,
                                                    padding='max_length',
                                                    truncation=True,
                                                    add_special_tokens=True
                                                    )
            input_ids.append(encode_dict['input_ids'])
            attention_masks.append(encode_dict['attention_mask'])
            labels.append([float(x) for x in [row['Geography'],row['Entities'],row['Time'],row['Narrative'],row['Overall'],row['Style'],row['Tone']]])
        
        return torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(labels)
    
    
    def data_loader(self, input_ids, attention_masks, labels):

        data = TensorDataset(input_ids, attention_masks, labels)
        loader = DataLoader(data, batch_size=self.config.batch_size, shuffle=True, drop_last=True)

        return loader

    def predict(self, model, data_loader):

        model.eval()
        test_pred, test_true = [], []
        with torch.no_grad():
            for idx, (ids, att, y) in enumerate(data_loader):
                y_pred = model(ids.to(self.device), att.to(self.device))
                y_pred = torch.squeeze(torch.add(torch.mul(y_pred[0], 0.5), torch.mul(y_pred[1], 0.5))).detach().cpu().numpy().tolist()
                y = y.squeeze().cpu().numpy().tolist()

                test_true.extend([x[4] for x in y])
                test_pred.extend([x[4] for x in y_pred])

            return test_true, test_pred


    def calculate_weighted_loss(self, y_pred, y, criterion):

        loss = 0.0
        for i in range(7):
            y_pred_i, y_i = y_pred[:, i], y[:, i]
            loss += criterion(y_pred_i, y_i) * self.config.dims_weights[i]
        return loss


    def train(self, model, train_loader, valid_loader, optimizer, schedule):

        best_pearson = 0.0
        criterion = nn.MSELoss()
        model.train()

        for i in range(self.config.epoch):
            start_time = time.time()
            train_loss_sum = 0.0
            
            logging.info(f"—————————————————————— Epoch {i+1} ——————————————————————")
            
            for idx, (ids, att, y) in enumerate(train_loader):

                ids, att, y = ids.to(self.device), att.to(self.device), y.to(self.device)
                y_pred1, y_pred2 = model(ids, att)
                y_pred1, y_pred2, y = torch.squeeze(y_pred1), torch.squeeze(y_pred2), torch.squeeze(y)

                loss1 = self.calculate_weighted_loss(y_pred1, y, criterion) * self.config.losses_weights['forward_weight']
                loss2 = self.calculate_weighted_loss(y_pred2, y, criterion) * self.config.losses_weights['forward_weight']
                loss_r = self.calculate_weighted_loss(y_pred1, y_pred2, criterion) * self.config.losses_weights['rdrop_weight']     
                loss = (loss1 + loss2 + loss_r) / self.config.gradient_acc

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                schedule.step()
                train_loss_sum += loss.item()

                if (idx+1) % (len(train_loader) // 10) == 0:
                    logging.info("Epoch {:02d} | Step {:03d}/{:03d} | Loss {:.4f} | Time {:.2f}".format(i+1, idx+1, len(train_loader), train_loss_sum/(idx+1), time.time()-start_time))
            

            logging.info("Start evaluating!")
            dev_true, dev_pred = self.predict(model, valid_loader)
            cur_pearson = np.corrcoef(dev_true, dev_pred)[0][1]
            
            if cur_pearson > best_pearson:
                best_pearson = cur_pearson
                torch.save(model.state_dict(), self.config.model_save_path)
            
            logging.info("Current dev pearson is {:.4f}, best pearson is {:.4f}".format(cur_pearson, best_pearson))
            logging.info("Time costed : {}s \n".format(round(time.time() - start_time, 3)))
    

    def run_finetune(self):

        train_loader = self.data_loader(*self.dataset(self.config.training_set_path))
        dev_loader = self.data_loader(*self.dataset(self.config.testing_set_path))

        model = MMRegressor(self.config.model_pretrain_dir).to(self.device)
        
        for param in model.parameters():
            param.requires_grad = True

        total_steps = len(train_loader) * self.config.epoch

        optimizer = AdamW(params=model.parameters(), 
                        lr=self.config.learning_rate, 
                        weight_decay=self.config.weight_decay)      
        schedule = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=self.config.warmup_rate*total_steps,
                                                    num_training_steps=total_steps)
        
        self.train(model, train_loader, dev_loader, optimizer, schedule)