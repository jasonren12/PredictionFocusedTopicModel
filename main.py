import argparse
import torch
from data_manager import load_Pang_Lee
from slda import sLDA
from pfslda import pfsLDA
from train import fit
from util import print_topics

def main():
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--K', type=int, default=5,
      help='Number of topics'
  )
  parser.add_argument(
      '--model', choices=['slda', 'pfslda'], default='slda',
      help='Specify which model to train'
  )
  parser.add_argument(
      '--p', type=float, default=0.10,
      help='Value for the switch prior for pf-sLDA'
  )
  parser.add_argument(
      '--alpha', type=bool, default=True,
      help='Specify if alpha is fixed'
  )
  parser.add_argument(
      '--path', type=str,
      help='Path to saved model to load before training'
  )
  parser.add_argument(
      '--lr', type=float, default=0.025,
      help='Initial learning rate'
  )
  parser.add_argument(
      '--lambd', type=float, default=0,
      help='Supervised task regularizer weight'
  )
  parser.add_argument(
      '--num_epochs', type=int, default=500,
      help='Number of epochs to train'
  )
  parser.add_argument(
      '--check', type=int, default=10,
      help='Number of epochs per stats check (print/save)'
  )
  parser.add_argument(
      '--batch_size', type=int, default=100,
  )
  parser.add_argument(
      '--y_thresh', type=float, default=None,
      help='Threshold for yscore (RMSE or AUC) to save model.'
  )
  parser.add_argument(
      '--c_thresh', type=float, default=None,
      help='Threshold for topic coherence to save model.'
  )


  args = parser.parse_args()
  
  # make sure args valid
  if args.K < 1:
    raise ValueError('Invalid number of topics.')
  
  p = args.p
  if p > 1 or p < 0:
    raise ValueError('Invalid switch prior p.')
  p = torch.tensor(p).to(device)
  p = torch.log(p / (1 - p))

  # load dataset and specify target type
  d = load_Pang_Lee()
  W = d['W']
  W_val = d['W_val']
  y = d['y']
  y_val = d['y_val']  
  W_test = d['W_test']
  y_test = d['y_test']
  vocab = d['vocab']
  version = 'real'

  V = W.shape[1]
  M = W.shape[0]
  M_val = W_val.shape[0]

  # instantiate model
  if args.model == 'slda':
      model = sLDA(args.K, V, M, M_val, args.alpha, device)
  elif args.model == 'pfslda':
      model = pfsLDA(args.K, V, M, M_val, p, args.alpha, device)
  model.to(device)

  # load saved model if path specified
  if args.path:
      state_dict = torch.load(args['path'], map_location=device)
      model.load_state_dict(state_dict)

  kwargs = {
      'W' : W,
      'y' : y, 
      'lr' : args.lr, 
      'lambd' : args.lambd,
      'num_epochs' : args.num_epochs, 
      'check' : args.check, 
      'batch_size' : args.batch_size, 
      'version' : version,
      'W_val' : W_val,
      'y_val' : y_val,
      'device' : device,
      'y_thresh' : args.y_thresh,
      'c_thresh' : args.c_thresh
  }

  fit(model, **kwargs)
  print_topics(model, 10, vocab)


if __name__ == '__main__':
  main()


