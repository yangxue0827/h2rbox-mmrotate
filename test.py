import numpy as np

def weighted_cross_entropy(x, y, w):
    max_x = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - max_x)
    sum_x = np.sum(e_x, axis=1, keepdims=True)
    softmax_x = e_x / sum_x

    dim = np.max(y) + 1
    dim = x.shape[1]
    onehot = np.eye(dim)[y]
    item1 = np.log(softmax_x) * onehot
    item2 = np.log(1 - softmax_x) * (1 - onehot)
    loss = -1 * (item1 + item2) * np.expand_dims(w, axis=1)
    # loss = np.sum(loss)  # 0.4229946107384874
    loss = np.mean(np.sum(loss, axis=1))  # 0.21149730536924366
    return loss

x = np.array([[4, 2.5, 1], [3, -2, 7]]) # logits of 2 samples
y = np.array([0, 2]) # ground-truth labels
w = np.array([0.8, 1.2]) # weights of 2 samples
loss = weighted_cross_entropy(x, y, w) # output loss
print(loss) # output loss=?
