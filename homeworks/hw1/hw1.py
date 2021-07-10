#%%

from numpy.lib.arraysetops import isin
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data.dataloader import DataLoader
from deepul.hw1_helper import *

#%%

train_data = np.random.randint(low =0 , high = 100, size = (1000,1))
train_loader = data.DataLoader(train_data, batch_size = 128, shuffle = True)
for x in train_loader:
    print(x)

L = torch.ones(10)
print(L)
print(L.unsqueeze(0)) # adds a dimension to the input index
print(L.unsqueeze(0).repeat(20, 1))  # "tiles" the tensor with i rows and j cols

#%% 1. A)


def train(model, train_loader, optimizer, epoch):
    model.train()
    train_losses = []
    for x in train_loader:
        x = x.cuda().contiguous()
        loss = model.loss(x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    return train_losses

def eval_loss(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x in data_loader:
            x = x.cuda().contiguous()
            loss = model.loss(x)
            total_loss += loss * x.shape[0]
        avg_loss = total_loss/ len(data_loader.dataset)
    # nats/dim => divide by ln(e) = 1
    return avg_loss.item()

def train_epochs(model, train_loader, test_loader, train_args):
    epochs, lr = train_args["epochs"], train_args['lr']
    optimizer = optim.Adam(model.parameters(), lr = lr)
    train_losses = []
    test_losses = []
    for epoch in range(epochs):
        model.train()
        train_losses.extend(train(model, train_loader, optimizer, epoch))
        test_loss = eval_loss(model, test_loader)
        test_losses.append(test_loss)
        print(f"Epoch {epoch}, Test loss {test_loss:.4f}")
    return train_losses, test_losses

#TODO(agro): is there another way to do this?
class Histogram(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.logits = nn.Parameter(torch.zeros(d), requires_grad= True)

    def loss(self, x):
        # This cross entropy does: Softmax(logits) and then lakes the 
        # negative log likeihood with the target x.
        # unsqueeze adds a dimension in index 1. repeat tiles a matrix
        # with x.shape[0] rows and 1 column with the matrix logits
        return F.cross_entropy(
            self.logits.unsqueeze(0).repeat(x.shape[0], 1), x.long()
        )

    @property
    def distribution(self):
        return F.softmax(self.logits, dim = 0).detach().cpu().numpy()

def q1_a(train_data, test_data, d, dset_id):
    """
    train_data: An (n_train,) numpy array of integers in {0, ..., d-1}
    test_data: An (n_test,) numpy array of integers in {0, .., d-1}
    d: The number of possible discrete values for random variable x
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
                used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (d,) of model probabilities

    """

    model = Histogram(d).cuda()
    train_loader = data.DataLoader(train_data, batch_size = 128, shuffle =True)
    test_loader = data.DataLoader(test_data, batch_size = 128)
    train_losses, test_losses = train_epochs(
        model, train_loader, test_loader, dict(epochs = 20, lr = 1e-1)
    )
    return train_losses, test_losses, model.distribution

#%%

q1_save_results(1, "a", q1_a)

# %%

q1_save_results(2, "a", q1_a)
# %% 1. B)


class MixLogistics(nn.Module):

    def __init__(self, d, n = 4):
        super().__init__()
        self.d = d
        self.n = 4


        # log of pi's
        self.logits = nn.Parameter(torch.zeros(n), requires_grad=True)
        self.means = nn.Parameter(torch.linspace(0, d, 4), requires_grad= True)
        # why do we do log of slopes?
        self.log_of_slopes = nn.Parameter(torch.randn(n), requires_grad=True)

    def forward(self, x):
        # tile x into a tensor batch size x n
        x = x.float().unsqueeze(1).repeat(1, self.n) 
        means = self.means.unsqueeze(0)
        inv_slopes = torch.exp(self.log_of_slopes.unsqueeze(0))
        cdf_p = torch.sigmoid(inv_slopes*(x + 0.5 - means))
        cdf_m = torch.sigmoid(inv_slopes*(x - 0.5 - means))
        cdf_d = cdf_p - cdf_m
        x_log_probs = torch.log(torch.clamp(cdf_d, min = 1e-12))

        # if all x's were 0, what would the cdf value be:

        log_cdf_min = torch.log(
            torch.clamp(
                torch.sigmoid(
                    inv_slopes * (0.5 - means)
                ),
                min = 1e-12
            )
        )

        log_cdf_max = torch.log(
            torch.clamp(
                1 - torch.sigmoid(
                    inv_slopes * (self.d - 1 -0.5 - means)
                ),
                min = 1e-12
            )
        )

        x_log_probs = torch.where(
            x < 1e-3, log_cdf_min, torch.where(
                x > self.d - 1 - 1e-3, log_cdf_max, x_log_probs
            )
        )

        pi_log_probs = F.log_softmax(self.logits, dim = 0).unsqueeze(0)
        log_probs = x_log_probs + pi_log_probs
        return torch.logsumexp(log_probs, dim = 1)


    def loss(self, x):
        return -torch.mean(self(x))

    @property
    def distribution(self):
        with torch.no_grad():
            x = torch.FloatTensor(np.arange(self.d)).cuda()
            distribution = self(x).exp()
        return distribution.detach().cpu().numpy()

def q1_b(train_data, test_data, d, dset_id):
    """
    train_data: An (n_train,) numpy array of integers in {0, ..., d-1}
    test_data: An (n_test,) numpy array of integers in {0, .., d-1}
    d: The number of possible discrete values for random variable x
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
            used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (d,) of model probabilities
    """
    
    """ YOUR CODE HERE """

    model = MixLogistics(d).cuda()
    train_loader = data.DataLoader(train_data, batch_size = 128, shuffle =True)
    test_loader = data.DataLoader(test_data, batch_size = 128)
    train_losses, test_losses = train_epochs(
        model, train_loader, test_loader, dict(epochs = 10, lr = 1e-1)
    )
    return train_losses, test_losses, model.distribution
# %%

q1_save_results(1, 'b', q1_b)
# %%

q1_save_results(2, "b", q1_b)
# %%2 A)

def to_one_hot(labels, d):
    one_hot = torch.FloatTensor(labels.shape[0], d).cuda()
    one_hot.zero_()
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias = True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("mask", torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)

class MADE(nn.Module):
    """
    fit a model for the probability of x1 ie. p(x1) 
    fit a model for the probability of x2 given x1 ie. p(x2 | x1)
    The output of the model is then p(x1, x2) = p(x2 | x1) p(x1)
    """

    def __init__(
        self,
        input_shape,
        d,
        hidden_size = [512, 512, 512],
        ordering = None,
        one_hot_input = False
    ):
        super().__init__()
        self.input_shape = input_shape
        self.nin = np.prod(input_shape)
        self.nout = self.nin * d
        self.d = d
        self.hidden_sizes = hidden_size
        self.ordering = np.arange(self.nin) if ordering is None else ordering
        self.one_hot_input = one_hot_input

        self.net = []
        hs = [self.nin *d if one_hot_input else self.nin] + self.hidden_sizes + [self.nout]
        for i in range(len(hs) - 1):
            h0 = hs[i]
            h1 = hs[i+1]
            self.net.extend([
                MaskedLinear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()
        self.m = {}
        self.create_mask()

    def create_mask(self):
        L = len(self.hidden_sizes)

        self.m[-1] = self.ordering
        # understand what is going on here?
        for l in range(L):
            self.m[l] = np.random.randint(
                self.m[l - 1].min(), self.nin -1, size = self.hidden_sizes[l]
            )
        masks = [
            self.m[l-1][:, None] <= self.m[l][None, :] for l in range(L)
        ]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])
        # for one hot
        masks[-1].repeat(masks[-1], self.d, axis = 1)
        if self.one_hot_input:
            masks[0] = np.repeat(masks[0], self.d, axis = 0)

        ind = 0
        for l in self.net.modules():
            if isinstance(l, MaskedLinear):
                l.set_mask(masks[ind])
                ind += 1

    def forward(self, x):
        batch_size = x.shape(0)
        if self.one_hot_input:
            x = x.long().contiguous().view(-1)
            x = to_one_hot(x, self.d)
            x = x.view(batch_size, -1)
        else:
            x = x.float()
            x = x.view(batch_size, self.nin)
        logits = self.net(x).view(batch_size, self.nin, self.d)
        return logits.permute(0, 2, 1).contiguous().view(batch_size, self.d, *self.input_shape)

    def loss(self, x):
        return F.cross_entropy(self(x), x.long())

    @property
    def distribution(self):
        assert self.input_shape == (2,)
        x = np.mgrid[0:self.d, 0:self.d].reshape(2, self.d **2).T
        x = torch.LongTensor(x).cuda()
        log_probs = F.log_softmax(self(x), dim = 1)
        distribution = torch.gather(log_probs, 1, x.unsqueeze(1)).squeeze(1)
        distribution = distribution.sum(dim = 1)
        return distribution.exp().view(self.d, self.d).detach().cpu().numpy()

def q2_a(train_data, test_data, d, dset_id):
    """
    train_data: An (n_train, 2) numpy array of integers in {0, ..., d-1}
    test_data: An (n_test, 2) numpy array of integers in {0, .., d-1}
    d: The number of possible discrete values for each random variable x1 and x2
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
            used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (d, d) of probabilities (the learned joint distribution)
    """
    
    """ YOUR CODE HERE """

    model = MADE((2,), d, hidden_size = [100, 100], one_hot_input = True).cuda()
    train_loader = data.DataLoader(train_data, batch_size = 128, shuffle = True)
    test_loader = data.DataLoader(test_data, batch_size = 128)
    train_losses, test_losses = train_epochs(
        model, train_loader, test_loader, dict(epochs = 10, lr = 1e-1)
    )
    return train_losses, test_losses, model.distribution

#%%

q2_save_results(1, 'a', q2_a)
# %%
