import torch
import numpy as np
from util import batchify, coherence


def fit(
    model,
    W,
    y,
    lr,
    lambd,
    num_epochs,
    batch_size,
    check,
    version,
    W_val,
    y_val,
    device,
    y_thresh,
    c_thresh,
):
    """
    Fit model on W (count data), y (targets) w/:
        - lr: initial learning rate
        - lambd: supervised task regularizer weight
        - num_epochs and batch_size
        - version: specifies modeling targets as Normal (real)
                   or Benoulli (binary)
  
    Every check epochs, calc and print topic coherence, val yscore
    (based on W_val, and y_val). If y_thresh and c_thresh specified,
    save model when val yscore or coherence are better than their 
    thresholds. O/w save final model after num_epochs.
    """
    print(f"Training {model.name} on {device}.")

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for i in range(num_epochs):
        # batch necessary parts of data
        to_batch = [W, model.phi, model.gamma, y]
        batches = batchify(to_batch, batch_size)
        W_b, phi_b, gamma_b, y_b = batches[0], batches[1], batches[2], batches[3]

        tot = 0
        for j in range(len(W_b)):
            opt.zero_grad()
            elbo = model.ELBO(
                W_b[j].to(device),
                phi_b[j],
                gamma_b[j],
                y_b[j].to(device),
                version=version,
            )
            tot += elbo.item()
            loss = -1 * elbo + lambd * (model.eta ** 2).sum()
            loss.backward()
            opt.step()

        if i % check == 0:
            val_yscore, c = calc_stats_and_print(
                model, W, W_val.to(device), y_val.to(device), tot / W.sum(), i, version
            )

            save = False
            if (y_thresh and val_yscore < y_thresh) or (c_thresh and c > c_thresh):
                save = True
            if save:
                path = f"models/{model.name}_ y{val_yscore:.2f}_c{c:.2f}.pt"
                torch.save(model.state_dict(), path)

    # save last model if no thresholds
    if not y_thresh and not c_thresh:
        val_yscore, c = calc_stats_and_print(
            model,
            W,
            W_val.to(device),
            y_val.to(device),
            tot / W.sum(),
            num_epochs,
            version,
        )
        path = f"models/{model.name}_ y{val_yscore:.2f}_c{c:.2f}.pt"
        torch.save(model.state_dict(), path)

    return


def yscore(model, W, y, version):
    _, preds = model.pred(W)

    # RMSE
    if version == "real":
        score = ((preds - y) ** 2).mean().sqrt()

    # AUC
    elif version == "binary":
        probs = preds.sigmoid().cpu().detach().numpy()
        score = auc(y.cpu().detach().numpy(), probs)

    else:
        raise ValueError("Invalid Version. Expected real or binary.")

    return score


def calc_stats_and_print(model, W, W_val, y_val, elbo, i, version):
    val_yscore = yscore(model, W_val, y_val, version)
    beta = model.beta.softmax(dim=1).cpu().detach().numpy()
    topk = np.argsort(beta, axis=1)[:, -50:]
    c = coherence(topk, W)
    print(f"Epoch: {i}")
    print(f"ELBO: {elbo}")
    print(f"Val yscore: {val_yscore}")
    print(f"Coherence: {c}\n")
    return val_yscore, c
