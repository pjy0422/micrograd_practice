import random

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


def prepare_data():
    with open("names.txt", "r") as f:
        data = f.read().splitlines()
    ch_list = ["."] + sorted(list(set("".join(data))))

    stoi = {ch: i for i, ch in enumerate(ch_list)}
    itos = {i: ch for i, ch in enumerate(ch_list)}
    return data, stoi, itos


def split_data(data):
    random.shuffle(data)
    train_data = data[: int(0.8 * len(data))]
    dev_data = data[int(0.8 * len(data)) : int(0.9 * len(data))]
    test_data = data[int(0.9 * len(data)) :]

    return train_data, dev_data, test_data


def get_input(data, stoi, itos):
    xs = []
    ys = []
    N = torch.zeros((27, 27, 27))
    for word in data:
        chs = ["."] + list(word) + ["."]
        for ch1, ch2, ch3 in zip(chs, chs[1:], chs[2:]):
            xs.append((stoi[ch1], stoi[ch2]))
            ys.append(stoi[ch3])
            N[stoi[ch1], stoi[ch2], stoi[ch3]] += 1
    return torch.tensor(xs), torch.tensor(ys), N


def train(
    data,
    max_epochs=50,
    stoi=None,
    itos=None,
    alpha=50,
    lam=0.1,
    enc="onehot",
    loss="manual",
):
    g = torch.Generator().manual_seed(42)
    weight = torch.randn(2, 27, 27, requires_grad=True)
    xs, ys, N = get_input(data, stoi=stoi, itos=itos)
    for epoch in range(max_epochs):
        if enc == "onehot":
            xenc = F.one_hot(xs, num_classes=27).float()
        elif enc == "index":
            one_hot_enc = F.one_hot(xs, num_classes=27).float()
            xenc = torch.zeros((len(xs), 2, 27))
            xenc[torch.arange(len(xs)), 0, xs[:, 0]] = 1
            xenc[torch.arange(len(xs)), 1, xs[:, 1]] = 1
        logits = torch.einsum("ijk,jkl->il", xenc, weight)
        count = torch.exp(logits)
        probs = count / count.sum(dim=1, keepdim=True)
        if loss == "manual":
            tloss = (
                -probs[torch.arange(len(ys)), ys].log().mean()
                + lam * (weight**2).mean()
            )
        elif loss == "ce":
            tloss = F.cross_entropy(logits, ys) + lam * (weight**2).mean()
        weight.grad = None
        tloss.backward()
        weight.data -= alpha * weight.grad
        print(f"Epoch {epoch}, Train Loss {tloss.item():.3f}")
    return weight


def eval(data, stoi, itos, weight):
    xs, ys, _ = get_input(data, stoi=stoi, itos=itos)
    xenc = F.one_hot(xs, num_classes=27).float()
    logits = torch.einsum("ijk,jkl->il", xenc, weight)
    count = torch.exp(logits)
    probs = count / count.sum(dim=1, keepdim=True)

    tloss = -probs[torch.arange(len(ys)), ys].log().mean()
    print(f"Loss {tloss.item():.3f}")
    return tloss.item()


def e_01():
    data, stoi, itos = prepare_data()
    train(data, stoi=stoi, itos=itos)
    # Final loss is 2.263 which is better than bigram loss: 2.5, for 50 epochs


def e_02():
    data, stoi, itos = prepare_data()
    train_data, dev_data, test_data = split_data(data)
    weight = train(train_data, stoi=stoi, itos=itos, lam=0.01)
    eval(dev_data, stoi, itos, weight)
    eval(test_data, stoi, itos, weight)
    # Evaluate on dev and test data
    # Train loss is 2.291
    # Dev loss is 2.29
    # Test loss is 2.304
    # The loss is consistent across all the datasets, so the model is notoverfitted.


def e_03():
    data, stoi, itos = prepare_data()
    train_data, dev_data, test_data = split_data(data)
    lam_list = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    best_lam, best_loss = None, float("inf")
    for lam in lam_list:
        weight = train(train_data, stoi=stoi, itos=itos, lam=lam)
        dev_loss = eval(dev_data, stoi, itos, weight)
        if dev_loss < best_loss:
            best_loss = dev_loss
            best_lam = lam
    print(f"Best lambda: {best_lam}, Best Loss: {best_loss}")
    # Best lambda: 0.01, Best loss: 2.29


def e_04():
    data, stoi, itos = prepare_data()
    train_data, dev_data, test_data = split_data(data)
    weight = train(train_data, stoi=stoi, itos=itos, lam=0.01, enc="index")


def e_05():
    data, stoi, itos = prepare_data()
    train_data, dev_data, test_data = split_data(data)
    weight = train(train_data, stoi=stoi, itos=itos, lam=0.01, loss="ce")


if __name__ == "__main__":
    e_04()
