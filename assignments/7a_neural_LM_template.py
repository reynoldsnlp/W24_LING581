"""
Pytorch implementation of Bengio's Neural Probabilistic Language Model (NPLM)
using the Brown corpus (provided by NLTK).

Bengio's original paper: http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
"""

from collections import Counter
import multiprocessing
from pprint import pprint
import time

import nltk
from nltk.corpus import brown
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


def get_id_of_word(word, unk="<UNK>"):
    """Get id of given word (<UNK> if not found)."""
    unknown_word_id = word_to_id_mappings[unk]
    return word_to_id_mappings.get(word, unknown_word_id)


def get_train_dev(corpus, ngram_size=3):
    """Create training and dev sets"""
    x_train = []
    y_train = []
    x_dev = []
    y_dev = []

    for para_id, paragraph in enumerate(corpus.paras()):
        for sentence in paragraph:
            for i in range(len(sentence)):
                *x, y = sentence[i:i + ngram_size]
                if len(x) != ngram_size - 1:
                    # sentence boundary reached (ignore sentences < NGRAM_SIZE)
                    break
                # convert words to ids
                x = [get_id_of_word(word.lower()) for word in x]
                y = [get_id_of_word(y[0])]
                if para_id < TRAIN_PARAGRAPHS:
                    x_train.append(x)
                    y_train.append(y)
                else:
                    x_dev.append(x)
                    y_dev.append(y)
    return np.array(x_train), np.array(y_train), np.array(x_dev), np.array(y_dev)


class NPLM(nn.Module):
    """Neural Probabilistic Language Model (Bengio 2003)."""

    def __init__(self, vocab_size, embedding_dim, context_size, hidden_size):
        super().__init__()
        # TODO build neural network

    def forward(self, inputs):
        # TODO compute feedforward step and return log probabilities (use log_softmax)


def get_accuracy_from_log_probs(log_probs, labels):
    """Get accuracy from log probabilities"""
    probs = torch.exp(log_probs)
    predicted_label = torch.argmax(probs, dim=1)
    acc = (predicted_label == labels).float().mean()
    return acc


def evaluate(model, criterion, dataloader, gpu):
    """Evaluate model on dev data"""
    model.eval()

    mean_acc, mean_loss = 0, 0
    count = 0

    with torch.no_grad():
        dev_st = time.time()
        for it, data_tensor in enumerate(dataloader):
            context_tensor = data_tensor[:, 0:CONTEXT_SIZE]
            target_tensor = data_tensor[:, CONTEXT_SIZE]
            context_tensor, target_tensor = context_tensor.to(device), target_tensor.to(device)
            log_probs = model(context_tensor)
            mean_loss += criterion(log_probs, target_tensor).item()
            mean_acc += get_accuracy_from_log_probs(log_probs, target_tensor)
            count += 1
            if it % 500 == 0:
                print(f"Dev Iteration {it} complete. "
                      f"Mean Loss: {mean_loss / count}; "
                      f"Mean Acc:{mean_acc / count}; "
                      f"Time taken (s): {time.time()-dev_st}")
                dev_st = time.time()

    return mean_acc / count, mean_loss / count


if __name__ == "__main__":
    nltk.download("brown")

    UNK_symbol = "<UNK>"
    gpu = 0
    NGRAM_SIZE = 3
    CONTEXT_SIZE = NGRAM_SIZE - 1
    EMBEDDING_DIM = 100
    HIDDEN_SIZE = 100
    BATCH_SIZE = 256
    EPOCHS = 5
    torch.manual_seed(42)


    ### Build vocabulary ###

    # training set: first 12000 paragraphs (dev set: remaining 3000+)
    TRAIN_PARAGRAPHS = 12000

    train_freq = Counter()
    for i, paragraph in enumerate(brown.paras()):
        if i == TRAIN_PARAGRAPHS:
            break
        for sentence in paragraph:
            train_freq.update(word.lower() for word in sentence)
    print(train_freq.most_common(200))

    # create vocabulary
    vocab = set([UNK_symbol])
    for word, freq in train_freq.items():
        if freq >= 5:
            vocab.add(word)

    print("Vocab length:", len(vocab))

    word_to_id_mappings = {}
    for i, word in enumerate(vocab):
        word_to_id_mappings[word] = i


    ### Create DataLoaders ###

    # check if gpu is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    available_workers = multiprocessing.cpu_count()

    x_train, y_train, x_dev, y_dev = get_train_dev(brown, ngram_size=3)
    print("Train shape of x and y:", x_train.shape, y_train.shape)
    print("Dev shape of x and y:", x_dev.shape, y_dev.shape)


    print(f"--- Creating training and dev dataloaders with {BATCH_SIZE} batch size ---")
    train_set = np.concatenate((x_train, y_train), axis=1)
    dev_set = np.concatenate((x_dev, y_dev), axis=1)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=available_workers)
    dev_loader = DataLoader(dev_set, batch_size=BATCH_SIZE, num_workers=available_workers)


    # ------------------------- TRAIN & SAVE MODEL ------------------------

    # Using negative log-likelihood loss
    loss_function = nn.NLLLoss()

    # create model
    model = NPLM(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE, HIDDEN_SIZE)

    model.to(device)

    # using ADAM optimizer
    optimizer = optim.Adam(model.parameters(), lr=2e-3)

    best_acc = 0
    best_model_path = None
    for epoch in range(EPOCHS):
        print(f"\n--- Training model Epoch: {epoch+1} ---")
        for it, data_tensor in enumerate(train_loader):
            st = time.time()  # start time  (use time.time() - st to get time taken per minibatch)
            context_tensor = data_tensor[:, 0:CONTEXT_SIZE]
            target_tensor = data_tensor[:, CONTEXT_SIZE]
            context_tensor, target_tensor = context_tensor.to(device), target_tensor.to(device)

            # TODO Write the training loop
            # TODO At the end of each 500 mini-batches...
            #          * ...print the batch number and loss
        # TODO At the end of each epoch...
        #          * ...evaluate the model on the dev set and print results
        #          * ...save the model to a file
        #          * ...keep track of the best model's path in best_model_path


    # ---------------------- Loading Best Model -------------------

    best_model = NPLM(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE, HIDDEN_SIZE)
    best_model.load_state_dict(torch.load(best_model_path))
    best_model.to(device)

    cos = nn.CosineSimilarity(dim=0)

    lm_similarities = {}
    # word pairs to calculate similarity
    words = [
             ('computer', 'keyboard'),
             ('war', 'peace'),
             ('war', 'about'),
             ('school', 'college'),
             ('school', 'who'),
             ('national', 'international'),
             ('national', 'few'),
            ]

    # ----------- Calculate LM similarities using cosine similarity ----------

    for word_pair in words:
        w1, w2 = word_pair
        words_tensor = torch.LongTensor([get_id_of_word(w1), get_id_of_word(w2)])
        words_tensor = words_tensor.to(device)
        # get word embeddings from the best model
        words_embeds = best_model.embeddings(words_tensor)
        # calculate cosine similarity between word vectors
        sim = cos(words_embeds[0], words_embeds[1])
        lm_similarities[word_pair] = sim.item()

    print("\n\nSimilarity scores:")
    pprint(lm_similarities)
