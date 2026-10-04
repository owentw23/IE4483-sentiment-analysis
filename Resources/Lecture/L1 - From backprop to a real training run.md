---
tags: [resources, lecture]
status: developing
created: 2026-10-05
modified: 2026-10-05
source: "EE4483 - Neural Networks and Backpropagation.pdf (repo root); sources below"
---
# L1 — From backprop to a real training run

Part of [[MOC - ML Training Lecture]].

## 1. What your EE4483 lecture covers (summary)

**Perceptron.** $z = \mathbf{w}^\top\mathbf{x} + b$, $y = \text{step}(z)$. The decision boundary $\mathbf{w}^\top\mathbf{x}+b=0$ is a hyperplane, so a perceptron is a *linear classifier*. It can do AND and OR, but not XOR, because no single line separates XOR's classes.

**MLP.** Stack layers: each unit computes $net_j = \sum_i w_{ji} o_i + b_j$ and then $o_j = f(net_j)$. A hidden layer of OR and AND units solves XOR. The nonlinearity $f$ is essential: a stack of purely linear layers collapses into a single linear map. With enough hidden units, an MLP can approximate any continuous function (Hornik 1989). Being *able* to represent a function doesn't mean training will find it.

**Activations.**

| Activation | Formula | Range | Derivative |
|---|---|---|---|
| Sigmoid | $\sigma(z) = 1/(1+e^{-z})$ | (0, 1) | $\sigma(1-\sigma)$ |
| tanh | $\tanh z$ | (−1, 1) | $1-\tanh^2$ |
| ReLU | $\max(0,z)$ | [0, ∞) | 0 or 1 |

Both sigmoid and tanh *saturate*: at large $|z|$ the derivative is near 0, so learning stalls. A ReLU unit with $z<0$ passes no gradient, so it is "inactive" for that input.

**Forward inference.** With weights fixed, compute layer by layer. No target and no update are involved.

**Training.** The lecture's training steps:
1. **Loss.** The lecture uses squared error $E = \tfrac12\sum_k (t_k - o_k)^2$.
2. **Gradient descent.** Update $w \leftarrow w - \eta\,\partial E/\partial w$. The learning rate $\eta$ is too big if updates overshoot (the lecture's ReLU example: loss *rose* from 0.0050 to 0.0115) and too small if progress is slow.
3. **Backprop.** In the lecture's sign convention $\delta = -\partial E/\partial net$:
   - Output units: $\delta_k = (t_k - o_k) f'(net_k)$.
   - Hidden units: $\delta_j = f'(net_j)\sum_k w_{kj}\delta_k$.
   - Updates: $\Delta w_{ji} = \eta\,\delta_j o_i$ and $\Delta b_j = \eta\,\delta_j$.
4. **Update order.** Compute *all* deltas with the old weights, then update everything.
5. **When to update.**
   - **Stochastic:** one example per update.
   - **Mini-batch:** average the gradient over a small batch, then update.
   - **Full batch:** the whole training set per update.

   An **epoch** is one pass through the training set.
6. **When to stop.** Stop when the loss reaches a threshold, the updates become tiny, you hit an epoch cap, or (the one that matters) **a validation criterion signals overfitting**. Low training error on its own proves nothing about generalisation.

Work through the lecture's worked examples by hand once; they're good practice. In real projects you never compute δ by hand, as §4 explains.

## 2. Gap 1: classification uses cross-entropy, not squared error

The lecture uses squared error for simplicity. Real classifiers use **cross-entropy (CE)**, which is the negative log-probability the model gives to the correct class.

**Binary case.** Sigmoid output $o = \sigma(z)$ and target $t\in\{0,1\}$:
$$E = -[\,t\log o + (1-t)\log(1-o)\,]$$

**Multi-class case.** Softmax over logits $z_c$, so $p_c = e^{z_c}/\sum_{c'} e^{z_{c'}}$, and $E = -\log p_y$. Our project uses 2 logits and softmax, which is equivalent to the binary case.

**Why CE.** Do the lecture's output-delta derivation with CE instead of squared error:
$$\frac{\partial E}{\partial o} = -\frac{t}{o} + \frac{1-t}{1-o} = \frac{o-t}{o(1-o)}, \qquad \frac{\partial o}{\partial z} = o(1-o)$$
$$\Rightarrow\; \delta = -\frac{\partial E}{\partial z} = t - o$$

The $o(1-o)$ factor **cancels**:
- With squared error, $\delta = (t-o)\,o(1-o)$. When the output is confidently *wrong* (say $o\approx 0.001$ but $t=1$), $o(1-o)\approx 0$, so the network barely learns from its worst mistakes.
- With CE, the error signal is simply "how wrong you are".

For softmax with CE, the same result holds per logit: $\partial E/\partial z_c = p_c - y_c$.

CE also has a probabilistic meaning: minimising it is **maximum likelihood**, which makes the predicted probabilities as likely as possible to have produced the observed labels.

**Useful number.** A 2-class model that hasn't learned anything predicts 0.5/0.5, so its CE is $-\ln 0.5 = \ln 2 \approx 0.693$. If your first training step reports a loss far from 0.693, something is broken (see L5).

**Class-weighted CE**, which we use, multiplies each example's loss by a weight $w_y$ for its class: $E = -w_y\log p_y$. L2 covers why.

## 3. Gap 2: optimizers beyond plain gradient descent

The lecture's update $w \leftarrow w - \eta g$ (with $g$ the mini-batch gradient) is **SGD**. Practical improvements:
- **Momentum.** Keep a running average of past gradients, $v \leftarrow \beta v + g$, then update $w \leftarrow w - \eta v$. This smooths noisy mini-batch gradients and speeds progress along consistent directions.
- **Adam.** Keep running averages of $g$ (first moment $m$) and $g^2$ (second moment $v$). Update with $w \leftarrow w - \eta\,\hat m/(\sqrt{\hat v}+\epsilon)$. Each parameter gets its own effective step size: parameters with large, noisy gradients take smaller steps. Adam is robust to the choice of $\eta$ and is the default for transformers.
- **AdamW.** Adam with **decoupled weight decay**: shrink weights a little each step, $w \leftarrow w - \eta\lambda w$, separately from the gradient update. Weight decay is a regulariser (L5). Adding an L2 penalty to the loss *inside* Adam is not equivalent, because Adam's per-parameter scaling distorts it; Loshchilov & Hutter showed the decoupled version works better. **AdamW is the standard optimizer for fine-tuning transformers.**
- **Memory cost.** Adam stores two extra numbers ($m$, $v$) per parameter. For a 400M-parameter model that's 800M extra floats, which is why L6 covers 8-bit optimizers.

## 4. Gap 3: autograd, so you never hand-derive δ again

Frameworks like PyTorch record every operation in the forward pass (the *computational graph*). Calling `loss.backward()` then applies the chain rule automatically, which is exactly the lecture's δ recursion, generalised to any graph. Your job reduces to:
- defining the forward computation
- choosing the loss
- choosing the optimizer

## 5. The training loop: what "training a model" actually means

```python
model = MyModel()                      # random init (or pretrained weights)
opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

for epoch in range(num_epochs):
    model.train()                       # enables dropout etc.
    for x, y in train_loader:           # mini-batches, shuffled each epoch
        logits = model(x)               # forward pass
        loss = loss_fn(logits, y)       # compare with targets
        opt.zero_grad()                 # clear old gradients
        loss.backward()                 # backprop: fills .grad for every weight
        opt.step()                      # update: w <- w - lr * (AdamW rule)
    model.eval()                        # disables dropout
    with torch.no_grad():               # inference only, no graph
        val_metrics = evaluate(model, val_loader)
    save_checkpoint_if_best(val_metrics)  # "a validation criterion" from the lecture
```

Every model in this project, from logistic regression to the 400M-parameter transformer, is this loop. The only differences are what `MyModel` is, what `x` contains (L3) and the hyperparameters (L5). Hugging Face's `Trainer` is this loop with logging, mixed precision and checkpointing built in.

> [!question]- Check yourself 1: why does a sigmoid + squared-error network learn slowly when it is confidently wrong, and why doesn't sigmoid + CE?
> With squared error, $\delta=(t-o)\,o(1-o)$. When $o$ is near 0 or 1 the factor $o(1-o)\to 0$, so even a huge error gives a tiny gradient. With CE the factor cancels and $\delta = t-o$, which is largest exactly when the model is most wrong.

> [!question]- Check yourself 2: your 2-class classifier's loss at step 0 is 3.2. What does that suggest?
> Expected ≈ ln 2 ≈ 0.693 for an untrained model. A loss of 3.2 means the model starts with confident, wrong predictions. Likely causes: bad initialisation or scaling of the head, wrong labels or mapping (for example, labels not in {0, 1}), or the loss being applied to probabilities instead of logits (`CrossEntropyLoss` expects raw logits).

> [!question]- Check yourself 3: what are the three things you must choose to train any model in PyTorch?
> The forward computation (model architecture), the loss function, and the optimizer with its hyperparameters. Autograd handles the gradients.

## Sources and further study

- Nielsen, *Neural Networks and Deep Learning*, ch. 3 (cross-entropy vs quadratic cost, with exactly this derivation): http://neuralnetworksanddeeplearning.com/chap3.html
- 3Blue1Brown, backpropagation videos (visual companion to the lecture): https://www.3blue1brown.com/lessons/backpropagation and https://www.3blue1brown.com/lessons/backpropagation-calculus
- CS231n, *Linear classification* (softmax and cross-entropy): https://cs231n.github.io/linear-classify/
- D2L, *Softmax regression*: https://d2l.ai/chapter_linear-classification/softmax-regression.html · Optimisation chapter: https://d2l.ai/chapter_optimization/index.html
- Ruder, *An overview of gradient descent optimization algorithms*: https://arxiv.org/abs/1609.04747
- Loshchilov & Hutter, *Decoupled Weight Decay Regularization* (AdamW): https://arxiv.org/abs/1711.05101
- Karpathy, *Neural Networks: Zero to Hero* (builds autograd from scratch; best if you like code): https://karpathy.ai/zero-to-hero.html
