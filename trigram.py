import numpy as np
import torch
import torch.nn.functional as F

"""
Exercises:
E01: train a trigram language model, i.e. take two characters as an input to predict the 3rd one. Feel free to use either counting or a neural net. Evaluate the loss; Did it improve over a bigram model?
E02: split up the dataset randomly into 80% train set, 10% dev set, 10% test set. Train the bigram and trigram models only on the training set. Evaluate them on dev and test splits. What can you see?
E03: use the dev set to tune the strength of smoothing (or regularization) for the trigram model - i.e. try many possibilities and see which one works best based on the dev set loss. What patterns can you see in the train and dev set loss as you tune this strength? Take the best setting of the smoothing and evaluate on the test set once and at the end. How good of a loss do you achieve?
E04: we saw that our 1-hot vectors merely select a row of W, so producing these vectors explicitly feels wasteful. Can you delete our use of F.one_hot in favor of simply indexing into rows of W?
E05: look up and use F.cross_entropy instead. You should achieve the same result. Can you think of why we'd prefer to use F.cross_entropy instead?
E06: meta-exercise! Think of a fun/interesting exercise and complete it.


"""


with open("names.txt", "r") as f:
    data = f.read().splitlines()
ch_list = ["."] + sorted(list(set("".join(data))))

stoi = {ch: i for i, ch in enumerate(ch_list)}
itos = {i: ch for i, ch in enumerate(ch_list)}

xs = []
ys = []
N = torch.zeros((27, 27, 27))
for word in data:
    chs = ["."] + list(word) + ["."]
    for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
        xs.append((stoi[ch1], stoi[ch2]))
        ys.append(stoi[ch3])
        N[stoi[ch1], stoi[ch2], stoi[ch3]] += 1

xs = torch.tensor(xs)
ys = torch.tensor(ys).float()
print(xs.shape, ys.shape)
g = torch.Generator().manual_seed(42)
print(xs.shape, ys.shape)
print(xs[0].shape)
weight = torch.randn(2, 27, 27, requires_grad=True)

max_epochs = 100
loss = 0.0
for epoch in range(max_epochs):
    xenc = F.one_hot(xs, num_classes=27 * 27).float()
    logits = xenc.view(-1, 2 * 27 * 27) @ weight.view(2 * 27 * 27, 1)

    weight.grad = None
    loss.backward()
    weight.data += -3 * weight.grad
    log_likelihood = loss.item()
    print(f"Epoch {epoch}: Loss: {log_likelihood}")
