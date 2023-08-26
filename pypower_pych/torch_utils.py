import torch

import warnings
warnings.filterwarnings("ignore")

def sparse(*args, **kwargs):
    if len(args) == 1: # sparse((values, crow_ccol))
        values, crow_ccol = args[0]
    elif len(args) == 2: # sparse(values, crow_ccol)
        if len(args[0]) != 2 and (not isinstance(args[1][0], int)):
            values, crow_ccol = args[0], args[1]
        else:
            return torch.sparse_coo_tensor(torch.stack([args[0][1][0], args[0][1][1]], dim=0),
                                           values=args[0][0], size=args[1]
                                           ).to_sparse(layout=torch.sparse_csr)
    else:
        raise ValueError("sparse() takes 1 or 2 positional arguments but {} were given".format(len(args)))

    # crow = crow_ccol[0]
    # ccol = crow_ccol[1]

    return torch.sparse_coo_tensor(torch.stack([crow_ccol[0], crow_ccol[1]], dim=0), values, (max(crow_ccol[0])+1, max(crow_ccol[1])+1)).to_sparse(layout=torch.sparse_csr)

def issparse(x):
    if isinstance(x, torch.Tensor):
        return x.layout == torch.sparse_coo or x.layout == torch.sparse_csr
    else:
        return False

def vstack(*args):
    if len(args) == 1 and isinstance(args[0], list):
        # 递归调用，将列表中的每个元素转换为张量
        return vstack(*args[0])
    elif args[-1] is 'csr':
        return vstack(*args[:-1])
    else:
        return torch.cat(args, dim=0)

def hstack(*args):
    if len(args) == 1 and isinstance(args[0], list):
        # 递归调用，将列表中的每个元素转换为张量
        return hstack(*args[0])
    elif args[-1] is 'csr':
        return vstack(args[:-1])
    else:
        return torch.cat(args, dim=1)


def find(x):
    return torch.nonzero(x).flatten()

if __name__ == '__main__':
    # test sparse function
    values = [1, 2, 3, 4, 5, 6]
    crow_ccol = [[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 2]]
    # il = range(len(values))
    print(sparse(values, crow_ccol).to_dense())