import torch

with open("words.txt", "r") as f:
    words = f.read().splitlines()
print(len(words))
len(words)

count = torch.zeros((27,27), dtype=torch.int32)

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        count[ix1, ix2] += 1

p = count
P = (count+1).float()
P = P/P.sum(1, keepdim=True)
g = torch.manual_seed(2147483647)
for i in range(10):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

log_likelihood = 0.0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1

print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')


#_____________________________________________________________#


xs, ys = [], []

for word in words:
    chs = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
n = len(xs)
xs = torch.tensor(xs)
ys = torch.tensor(ys)

gen = torch.manual_seed(2147483647)
W = torch.randn((27, 27), generator=gen, requires_grad=True)
for i in range(10):
  enc_x = torch.nn.functional.one_hot(xs, num_classes=27).float()
  logits = enc_x @ W
  counts = logits.exp()
  probs = counts / counts.sum(1, keepdims=True)
  loss = - probs[torch.arange(n),ys].log().mean()
  print(loss.item())
  W.grad = None
  loss.backward()
  W.data -= 10* W.grad

ge = torch.manual_seed(544787867)
for i in range(10):
    res = []
    ix = 0
    while True:
        x = torch.nn.functional.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = x @ W
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=ge).item()
        res.append(itos[ix])
        if ix == 0:
            break
    print(''.join(res))

