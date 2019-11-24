import torch
from opt_einsum import contract

# batch all iterables in to_batch in same random order
def batchify(to_batch, batch_size):
    M = to_batch[0].shape[0]
    rand = torch.randperm(M)
    for thing in to_batch:
        thing = thing[rand]

    i = 0
    out = [[] for thing in to_batch]

    while i + batch_size < M:
        for j, thing in enumerate(to_batch):
            out[j].append(thing[i : i + batch_size])
        i += batch_size
    for j, thing in enumerate(to_batch):
        out[j].append(thing[i:])
    return out


def coherence_single(w1, w2, W):
    eps = 0.01
    dw1 = W[:, w1] > 0
    dw2 = W[:, w2] > 0
    N = W.shape[0]

    dw1w2 = (dw1 & dw2).float().sum() / N + eps
    dw1 = dw1.float().sum() / N + eps
    dw2 = dw2.float().sum() / N + eps

    return dw1w2.log() - dw1.log() - dw2.log()


# calc coherence of topics based on W
# See appendix of https://arxiv.org/pdf/1910.05495.pdf for details
def coherence(topics, W):
    score = 0
    count = 0
    K, V = topics.shape[0], topics.shape[1]
    for i in range(K):
        topic = topics[i]
        for j1 in range(len(topic) - 1):
            for j2 in range(j1 + 1, len(topic)):
                score += coherence_single(topic[j1], topic[j2], W)
    return score / (K * V * (V - 1) / 2)


# prints top n most probable words in each topic of the model
def print_topics(model, n, vocab):
    beta = model.beta.softmax(dim=1).cpu().detach().numpy()
    topn = np.argsort(beta, axis=1)[:, -n:]
    for i in range(model.K):
        print(f"Topic {i}: eta = {model.eta[i]}\n {vocab[topn[i]]}")


# calc the term of the ELBO involving targets for sLDA/pfsLDA
# when targets modeled as normal
def s_term_normal(y_batch, gamma_batch, eta, delta, M):
    h = -0.5 * M * delta.log() - (y_batch ** 2).sum() / (2 * delta)
    g0 = gamma_batch.sum(dim=1, keepdim=True)
    g = gamma_batch / g0
    outer = contract("mi,mj->mij", g, g, backend="torch")
    EXtX = (-outer / (g0.unsqueeze(2) + 1) + outer).sum(dim=0) + torch.diag(
        (g / (g0 + 1)).sum(dim=0)
    )
    EX = g
    first = contract("m,k,mk->", y_batch, eta, EX, backend="torch")
    second = contract("k,kq,q->", eta, EXtX, eta, backend="torch")
    s_term = h + (2 * first - second) / (2 * delta)
    return s_term


# calc the term of the ELBO involving targets for sLDA/pfsLDA
# when targets modeled as bernoulli
def s_term_bernoulli(y_batch, gamma_batch, eta):
    g0 = gamma_batch.sum(dim=1, keepdim=True)
    g = gamma_batch / g0
    probs = contract("mk,k->m", g, eta, backend="torch").sigmoid()
    # to prevent overflows in log
    probs_cpy = probs
    if probs.min() <= 0:
        c = probs.min().detach()
        probs = probs - c + self.epsilon
    s_term1 = (y_batch * probs.log()).sum()
    probs = probs_cpy
    if probs.max() >= 1:
        c = probs.max().detach()
        probs = probs - (c - 1) - self.epsilon
    s_term2 = ((1 - y_batch) * (1 - probs).log()).sum()
    s_term = s_term1 + s_term2
    return s_term
