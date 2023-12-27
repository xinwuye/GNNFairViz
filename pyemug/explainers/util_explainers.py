import networkx as nx
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import argparse
import torch.optim as optim

class EmbeddingHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.embedding = output[0].detach().clone()

    def remove(self):
        self.hook.remove()

def parse_optimizer(parser):
    '''Set optimizer parameters'''
    opt_parser = parser.add_argument_group()
    opt_parser.add_argument('--opt', dest='opt', type=str,
            help='Type of optimizer')
    opt_parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str,
            help='Type of optimizer scheduler. By default none')
    opt_parser.add_argument('--opt-restart', dest='opt_restart', type=int,
            help='Number of epochs before restart (by default set to 0 which means no restart)')
    opt_parser.add_argument('--opt-decay-step', dest='opt_decay_step', type=int,
            help='Number of epochs before decay')
    opt_parser.add_argument('--opt-decay-rate', dest='opt_decay_rate', type=float,
            help='Learning rate decay ratio')
    opt_parser.add_argument('--lr', dest='lr', type=float,
            help='Learning rate.')
    opt_parser.add_argument('--clip', dest='clip', type=float,
            help='Gradient clipping.')

def build_optimizer(args, params, weight_decay=0.0):
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam': # default
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    return  optimizer


def build_optimizer_train(args, params, weight_decay=0.0):
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer


def neighborhoods(adj, n_hops, use_cuda=False): # identify which nodes are in the n hop neighborhood of each node, indicated by every row or column of the output adjacency matrix
    """Returns the n_hops degree adjacency matrix adj."""
    adj = torch.tensor(adj, dtype=torch.float)
    if use_cuda:
        adj = adj.cuda()
    hop_adj = power_adj = adj
    for i in range(n_hops - 1):
        power_adj = power_adj @ adj # matrix multiplication
        hop_adj = hop_adj + power_adj
        hop_adj = (hop_adj > 0).float()
    return hop_adj.cpu().numpy().astype(int)

def all_embeddings(model, x, adj):
    # hooks = [util.EmbeddingHook(layer) for layer in model.children()]
    hooks = []
    for layer in model.children():
        # check if layer is a nn.Module.loss
        if 'loss' not in str(type(layer)): # check if layer is a nn.Module.loss
            if isinstance(layer, nn.ModuleList):
                for sublayer in layer:
                    hooks.append(EmbeddingHook(sublayer))
            else:
                hooks.append(EmbeddingHook(layer))

    # Forward pass
    output = model(x, adj)
    # Access embeddings of each layer
    layer_embeddings = [hook.embedding for hook in hooks]
    for i, emb in enumerate(layer_embeddings):
        if emb.shape[0] == 1:
            # get rid of the batch dimension
            layer_embeddings[i] = emb.squeeze(0)
    # Remove hooks
    for hook in hooks:
        hook.remove()
    return layer_embeddings