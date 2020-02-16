import torch
from torch.autograd import Variable
from mlp import MLP
from utils import save_checkpoint, use_optimizer, use_cuda
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import numpy as np

class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self.modelA = MLP(config)
        self.modelB = MLP(config)
        if config['use_cuda'] is True:
            use_cuda(True, config['device_id'])
            self.modelA.cuda()
            self.modelB.cuda()
        print(self.modelA)
        if config['pretrain']:
            self.model.load_pretrain_weights()
        self.optA = use_optimizer(self.modelA, config)
        self.optB = use_optimizer(self.modelB, config)
        self.crit = torch.nn.MSELoss()
        self.alpha = config['alpha']

    def train_single_batch(self, book_user_embeddings, book_item_embeddings, book_rating,
                          movie_user_embeddings, movie_item_embeddings, movie_rating):
        self.optA.zero_grad()
        self.optB.zero_grad()
        book_ratings_pred1 = self.modelA(book_user_embeddings, book_item_embeddings)
        lossA1 = self.crit(book_ratings_pred1.squeeze(1), book_rating)
        book_ratings_pred2 = self.modelB(book_user_embeddings, book_item_embeddings, dual=True)
        lossA2 = self.crit(book_ratings_pred2.squeeze(1), book_rating)
        movie_ratings_pred1 = self.modelB(movie_user_embeddings, movie_item_embeddings)
        lossB1 = self.crit(movie_ratings_pred1.squeeze(1), movie_rating)
        movie_ratings_pred2 = self.modelA(movie_user_embeddings, movie_item_embeddings, dual=True)
        lossB2 = self.crit(movie_ratings_pred2.squeeze(1), movie_rating)
        lossA = (1-self.alpha)*lossA1 + self.alpha*Variable(lossA2.data, requires_grad=False)
        lossB = (1-self.alpha)*lossB1 + self.alpha*Variable(lossB2.data, requires_grad=False)
        lossA.backward(retain_graph=True)
        lossB.backward(retain_graph=True)
        orth_loss_A, orth_loss_B = torch.zeros(1), torch.zeros(1)
        reg = 1e-6
        for name, param in self.modelA.bridge.named_parameters():
            if 'bias' not in name:
                param_flat = param.view(param.shape[0], -1)
                sym = torch.mm(param_flat, torch.t(param_flat))
                sym -= torch.eye(param_flat.shape[0])
                orth_loss_A = orth_loss_A + (reg * sym.abs().sum())
        orth_loss_A.backward()
        for name, param in self.modelB.bridge.named_parameters():
            if 'bias' not in name:
                param_flat = param.view(param.shape[0], -1)
                sym = torch.mm(param_flat, torch.t(param_flat))
                sym -= torch.eye(param_flat.shape[0])
                orth_loss_B = orth_loss_B + (reg * sym.abs().sum())
        orth_loss_B.backward()
        self.optA.step()
        self.optB.step()
        if self.config['use_cuda'] is True:
            lossA = lossA.data.cpu().numpy()[0]
            lossB = lossB.data.cpu().numpy()[0]
            orth_loss_A = orth_loss_A.data.cpu().numpy()[0]
            orth_loss_B = orth_loss_B.data.cpu().numpy()[0]
        else:
            lossA = lossA.data.numpy()
            lossB = lossB.data.numpy()
            orth_loss_A = orth_loss_A.data.numpy()
            orth_loss_B = orth_loss_B.data.numpy()
        return lossA + lossB + orth_loss_A + orth_loss_B

    def train_an_epoch(self, train_book_loader, train_movie_loader, epoch_id):
        self.modelA.train()
        self.modelB.train()
        total_loss = 0
        for book_batch, movie_batch in zip(train_book_loader, train_movie_loader):
            assert isinstance(book_batch[0], torch.LongTensor)
            book_rating, book_user_embeddings, book_item_embeddings = Variable(book_batch[2]), Variable(book_batch[3]), Variable(book_batch[4])
            movie_rating, movie_user_embeddings, movie_item_embeddings = Variable(movie_batch[2]), Variable(movie_batch[3]), Variable(movie_batch[4])
            book_rating = book_rating.float()
            movie_rating = movie_rating.float()
            if self.config['use_cuda'] is True:
                book_rating = book_rating.cuda()
                movie_rating = movie_rating.cuda()
                book_user_embeddings = book_user_embeddings.cuda()
                book_item_embeddings = book_item_embeddings.cuda()
                movie_user_embeddings = movie_user_embeddings.cuda()
                movie_item_embeddings = movie_item_embeddings.cuda()
            loss = self.train_single_batch(book_user_embeddings, book_item_embeddings, book_rating,
                                           movie_user_embeddings, movie_item_embeddings, movie_rating)
            total_loss += loss

    def evaluate(self, evaluate_book_data, evaluate_movie_data, epoch_id):
        self.modelA.eval()
        self.modelB.eval()
        book_user, book_item, book_user_embeddings, book_item_embeddings, \
            book_golden = evaluate_book_data[0], evaluate_book_data[1], \
                Variable(evaluate_book_data[2]), Variable(evaluate_book_data[3]), evaluate_book_data[4]
        movie_user, movie_item, movie_user_embeddings, movie_item_embeddings, \
            movie_golden = evaluate_movie_data[0], evaluate_movie_data[1], \
                Variable(evaluate_movie_data[2]), Variable(evaluate_movie_data[3]), evaluate_movie_data[4]
        if self.config['use_cuda'] is True:
            book_user_embeddings = book_user_embeddings.cuda()   
            book_item_embeddings = book_item_embeddings.cuda()
            movie_user_embeddings = movie_user_embeddings.cuda()   
            movie_item_embeddings = movie_item_embeddings.cuda()
        book_scores = self.modelA(book_user_embeddings, book_item_embeddings)
        book_scores = book_scores.detach().numpy()
        movie_scores = self.modelB(movie_user_embeddings, movie_item_embeddings)
        movie_scores = movie_scores.detach().numpy()
        book_MSE = mean_squared_error(book_golden, book_scores)
        book_MAE = mean_absolute_error(book_golden, book_scores)
        movie_MSE = mean_squared_error(movie_golden, movie_scores)
        movie_MAE = mean_absolute_error(movie_golden, movie_scores)
        
        unique_book_user = list(set(book_user))
        unique_movie_user = list(set(movie_user))
        book_recommend, movie_recommend = [], []
        book_precision, movie_precision, book_recall, movie_recall = [], [], [], []
        for index in range(len(book_user)):
            book_recommend.append((book_user[index],book_item[index],book_golden[index],book_scores[index]))
        for index in range(len(movie_user)):
            movie_recommend.append((movie_user[index],movie_item[index],movie_golden[index],movie_scores[index]))      
        for user in unique_book_user:
            user_ratings = [x for x in book_recommend if x[0]==user]
            user_ratings.sort(key=lambda x:x[3], reverse=True)
            n_rel = sum((true_r >= 0.5) for (_, _, true_r, _) in user_ratings)
            n_rec_k = sum((est >= 0.5) for (_, _, _, est) in user_ratings)
            n_rel_and_rec_k = sum(((true_r >= 0.5) and (est >= 0.5))
                            for (_, _, true_r, est) in user_ratings)
            book_precision.append(n_rel_and_rec_k / n_rec_k if n_rec_k!=0 else 1)
            book_recall.append(n_rel_and_rec_k / n_rel if n_rel!=0 else 1)
        book_precision = np.mean(book_precision)
        book_recall = np.mean(book_recall)
        for user in unique_movie_user:
            user_ratings = [x for x in movie_recommend if x[0]==user]
            user_ratings.sort(key=lambda x:x[3], reverse=True)
            n_rel = sum((true_r >= 0.5) for (_, _, true_r, _) in user_ratings)
            n_rec_k = sum((est >= 0.5) for (_, _, _, est) in user_ratings)
            n_rel_and_rec_k = sum(((true_r >= 0.5) and (est >= 0.5))
                              for (_, _, true_r, est) in user_ratings)
            movie_precision.append(n_rel_and_rec_k / n_rec_k if n_rec_k!=0 else 0)
            movie_recall.append(n_rel_and_rec_k / n_rel if n_rel!=0 else 1)
        movie_precision = np.mean(movie_precision)
        movie_recall = np.mean(movie_recall)
        
        print('[Book Evluating Epoch {}] MSE = {:.4f}, MAE = {:.4f}, Precision = {}, Recall = {}'.format(epoch_id, book_MSE, book_MAE, book_precision, book_recall))
        print('[Movie Evluating Epoch {}] MSE = {:.4f}, MAE = {:.4f}, Precision = {}, Recall = {}'.format(epoch_id, movie_MSE, movie_MAE, movie_precision, movie_recall))
        
        return book_MSE, book_MAE, movie_MSE, movie_MAE

    def save(self, dirname, filename):
        with open(os.path.join(dirname, filename)+'A', 'wb') as f:
            torch.save(self.modelA.state_dict(), f)
        with open(os.path.join(dirname, filename)+'B', 'wb') as f:
            torch.save(self.modelB.state_dict(), f)