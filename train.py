import argparse
import pandas as pd
from engine import Engine
from data import SampleGenerator

parser = argparse.ArgumentParser('DDTCDR')
# Path Arguments
parser.add_argument('--num_epoch', type=int, default=100,help='number of epoches')
parser.add_argument('--batch_size', type=int, default=1024,help='batch size')
parser.add_argument('--lr', type=int, default=1e-2,help='learning rate')
parser.add_argument('--latent_dim', type=int, default=8,help='latent dimensions')
parser.add_argument('--alpha', type=int, default=0.03,help='dual learning rate')
parser.add_argument('--cuda', action='store_true',help='use of cuda')
args = parser.parse_args()

def dictionary(terms):
    term2idx = {}
    idx2term = {}
    for i in range(len(terms)):
        term2idx[terms[i]] = i
        idx2term[i] = terms[i]
    return term2idx, idx2term

mlp_config = {'num_epoch': args.num_epoch,
              'batch_size': args.batch_size,
              'optimizer': 'sgd',
              'lr': args.lr,
              'latent_dim': args.latent_dim,
              'nlayers':1,
              'alpha':args.alpha,
              'layers': [2*args.latent_dim,64,args.latent_dim],  # layers[0] is the concat of latent user vector & latent item vector
              'use_cuda': args.cuda,
              'pretrain': False,}

#Load Datasets
book = pd.read_csv('book.csv')
movie = pd.read_csv('movie.csv')
book['user_embedding'] = book['user_embedding'].map(eval)
book['item_embedding'] = book['item_embedding'].map(eval)
movie['user_embedding'] = movie['user_embedding'].map(eval)
movie['item_embedding'] = movie['item_embedding'].map(eval)

sample_book_generator = SampleGenerator(ratings=book)
evaluate_book_data = sample_book_generator.evaluate_data
sample_movie_generator = SampleGenerator(ratings=movie)
evaluate_movie_data = sample_movie_generator.evaluate_data

config = mlp_config
engine = Engine(config)
best_MSE = 1
for epoch in range(config['num_epoch']):
    print('Epoch {} starts !'.format(epoch))
    print('-' * 80)
    train_book_loader = sample_book_generator.instance_a_train_loader(config['batch_size'])
    train_movie_loader = sample_movie_generator.instance_a_train_loader(config['batch_size'])
    engine.train_an_epoch(train_book_loader, train_movie_loader, epoch_id=epoch)
    book_MSE, book_MAE, movie_MSE, movie_MAE = engine.evaluate(evaluate_book_data, evaluate_movie_data, epoch_id=epoch)