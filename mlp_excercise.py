"""
Exercises:
- E01: Tune the hyperparameters of the training to beat my best validation loss of 2.2
- E02: I was not careful with the intialization of the network in this video. (1) What is the loss you'd get if the predicted probabilities at initialization were perfectly uniform? What loss do we achieve? (2) Can you tune the initialization to get a starting loss that is much more similar to (1)?
- E03: Read the Bengio et al 2003 paper (link above), implement and try any idea from the paper. Did it work?
"""

import random

import torch
import torch.nn.functional as F

with open("names.txt", "r") as f:
    words = f.read().splitlines()
chars = sorted(list(set("".join(words))))
stoi = {c: i + 1 for i, c in enumerate(chars)}
stoi["."] = 0
itos = {i: c for c, i in stoi.items()}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataset(words, block_size=3):
    X, y = [], []
    for w in words:
        context = [0] * block_size
        for c in w + ".":
            ix = stoi[c]
            X.append(context)
            y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(y)


def train(X, y, lr=0.1, epochs=5000):
    g = torch.Generator(device=device).manual_seed(2147483647)
    C = torch.randn((27, 2), generator=g, requires_grad=True, device=device)
    W1 = torch.randn((6, 100), generator=g, requires_grad=True, device=device)
    b1 = torch.randn(100, generator=g, requires_grad=True, device=device)
    W2 = torch.randn((100, 27), generator=g, requires_grad=True, device=device)
    b2 = torch.randn(27, generator=g, requires_grad=True, device=device)
    parameters = [C, W1, b1, W2, b2]
    X, y = X.to(device), y.to(device)
    for epoch in range(epochs):
        # forward pass
        ix = torch.randint(0, len(X), (32,))
        emb = C[X[ix]]
        h = torch.tanh(emb.view(-1, 6) @ W1 + b1)
        logits = h @ W2 + b2
        loss = F.cross_entropy(logits, y[ix])
        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()
        # update
        for p in parameters:
            p.data -= lr * p.grad
        print(f"epoch {epoch}, loss {loss.item()}")


def split_dataset(words, p=0.8):
    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    X_train, y_train = build_dataset(words[:n1])
    X_val, y_val = build_dataset(words[n1:n2])
    X_test, y_test = build_dataset(words[n2:])
    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(words)
    train(X_train, y_train)


if __name__ == "__main__":
    main()
