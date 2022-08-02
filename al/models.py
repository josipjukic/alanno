from abc import ABC, abstractmethod
from functools import partial

from podium.datasets.dataset import Dataset
from podium.datasets.iterator import BucketIterator, Iterator

import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier as OVO
from sklearn.multiclass import OneVsRestClassifier as OVR
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC

from text.preprocessing import get_bert_vectorizer


RNN_TYPES = ["RNN", "LSTM", "GRU"]
TORCH_MODELS = ["mlp", "rnn", "lstm", "gru", "bert"]
RECURSIVE_MODELS = ["rnn", "lstm", "gru"]
TRANSFORMER_MODELS = ["bert"]


class TrainManager(ABC):
    """
    Class used to hold train/test data, and all hyperparameters needed for training a model.
    """

    @abstractmethod
    def __init__(self, sets, multilabel, *args, **kwargs):
        if len(sets) == 3:
            self.train, self.valid, self.test = sets
        else:
            self.train, self.test = sets

        self.train_size = len(self.train)
        self.test_size = len(self.test)
        self.multilabel = multilabel

    @abstractmethod
    def get_data(self, mode):
        pass

    @abstractmethod
    def get_partial_data(self, indices, mode):
        pass

    @abstractmethod
    def get_numpy_data(self, mode):
        pass

    def dataset_size(self, mode="train"):
        return len(getattr(self, mode))


class SklearnTM(TrainManager):
    def __init__(self, sets, multilabel):
        super().__init__(sets, multilabel)
        data_train = self.train.batch(add_padding=False)
        self.X_train, self.y_train = data_train.text, data_train.label
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)
        if not self.multilabel:
            self.y_train = self.y_train.ravel()

        if self.test_size > 0:
            data_test = self.test.batch(add_padding=False)
            self.X_test, self.y_test = data_test.text, data_test.label
            self.X_test = np.array(self.X_test)
            self.y_test = np.array(self.y_test)
        else:
            self.X_test = np.array([])
            self.y_test = np.array([])
        if not self.multilabel:
            self.y_test = self.y_test.ravel()

    def get_numpy_data(self, mode="train"):
        return self.get_data(mode)

    def get_data(self, mode="train"):
        X = getattr(self, f"X_{mode}")
        y = getattr(self, f"y_{mode}")
        return X, y

    def get_partial_data(self, indices, mode="train"):
        indices = np.array(indices, dtype=np.int)
        X, y = self.get_data(mode)
        return X[indices], y[indices]


class TorchTM(TrainManager):
    def __init__(
        self,
        sets,
        multilabel,
        criterion,
        optimizer=torch.optim.Adam,
        num_epochs=1,
        batch_size=64,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__(sets, multilabel)
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device

        data_train = self.train.batch(add_padding=True)
        self.X_train, self.y_train = data_train.text, data_train.label
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)

        if self.test_size > 0:
            data_test = self.test.batch(add_padding=True)
            self.X_test, self.y_test = data_test.text, data_test.label
            self.X_test = np.array(self.X_test)
            self.y_test = np.array(self.y_test)
        else:
            self.X_test = np.array([])
            self.y_test = np.array([])

        """
        if not self.multilabel:
            self.y_train = self.y_train.ravel()
            self.y_test = self.y_test.ravel()
        """

    def device_tensor(self, data):
        return torch.tensor(np.array(data)).to(self.device)

    def get_data(self, mode="train", shuffle=True):
        data = getattr(self, mode)
        return Iterator(
            data,
            batch_size=self.batch_size,
            matrix_class=self.device_tensor,
            shuffle=shuffle,
        )

    def get_partial_data(self, indices, mode="train", shuffle=True):
        set_ = getattr(self, mode)
        examples = [set_[i] for i in indices]
        data = Dataset(examples, fields=set_.fields)
        return Iterator(
            data,
            batch_size=self.batch_size,
            matrix_class=self.device_tensor,
            shuffle=shuffle,
        )

    def get_numpy_data(self, mode="train"):
        return getattr(self, f"X_{mode}"), getattr(self, f"y_{mode}")

    @staticmethod
    def text_len_sort_key(example):
        tokens = example["text"][1]
        return -len(tokens)

    @staticmethod
    def extract_fields(dataset):
        return dataset.text, dataset.label


class AbstractModel(ABC):
    @abstractmethod
    def predict_proba(self, batch):
        pass

    @abstractmethod
    def predict(self, batch):
        pass

    @abstractmethod
    def al_step(self, train_manager, **kwargs):
        pass


class ScikitModel(AbstractModel):
    def al_step(self, train_manager, **kwargs):
        X, y = train_manager.get_data()
        self.fit(X, y)


class TorchModel(AbstractModel):
    def __init__(self):
        super().__init__()
        self.multilabel = False

    def _predict_proba(self, X):
        self.eval()
        y_pred = self.forward(X)
        if self.output_dim == 1:
            y_pred = torch.sigmoid(y_pred)
            y_pred = torch.cat([1.0 - y_pred, y_pred], dim=1)
        elif self.multilabel:
            y_pred = torch.sigmoid(y_pred)
        else:
            y_pred = F.softmax(y_pred, dim=1)
        self.train()
        return y_pred

    def _predict(self, X):
        self.eval()

        """
        dataloader = torch.utils.data.DataLoader(X,
            batch_size=128, pin_memory=True, shuffle=True)

        prediction_list = []
        for i, batch in enumerate(dataloader):
            pred = self.forward(batch)
            prediction_list.extend(pred.cpu())

        y_pred = torch.stack(prediction_list)
        """

        y_pred = self.forward(X)
        if self.output_dim == 1 or self.multilabel:
            out = torch.as_tensor(
                y_pred > 0,
                dtype=torch.long,
                device=self.device,
            )
        else:
            out = torch.argmax(y_pred, dim=1)

        self.train()
        return out

    def predict_proba(self, X):
        if not isinstance(self, BertClassifier):
            X = torch.tensor(X, device=self.device)
        with torch.no_grad():
            y_pred = self._predict_proba(X)
            return y_pred.cpu().numpy()

    def predict(self, X):
        if not isinstance(self, BertClassifier):
            X = torch.tensor(X, device=self.device)
        with torch.no_grad():
            out = self._predict(X)
            return out.cpu().numpy()

    def evaluate(self, iterator, criterion):
        # TODO: add F1 metric
        self.eval()
        running_loss = 0
        running_acc = 0

        with torch.no_grad():
            for batch_index, batch in enumerate(iterator, 1):
                y = batch.label
                y_pred = self(batch.text)

                loss = criterion(y_pred.squeeze(), y)
                loss_t = loss.item()
                running_loss += (loss_t - running_loss) / batch_index

                acc_t = (y_pred == y).sum() / y_pred.shape[0]
                running_acc += (acc_t - running_acc) / batch_index

        self.train()
        return running_loss, running_acc

    def train_loop(self, iterator, criterion, optimizer, num_epochs):
        self.train()
        for _ in range(num_epochs):
            for batch in iterator:
                X = batch.text
                y = batch.label
                # 5 step training routine
                # ------------------------------------------

                # 1) zero the gradients
                optimizer.zero_grad()

                # 2) compute the output
                y_pred = self(X)
                if y_pred.shape[1] == 1 or self.multilabel:
                    y = y.float()

                # 3) compute the loss
                loss = criterion(y_pred.squeeze(), y.squeeze())

                # 4) use loss to produce gradients
                loss.backward()

                # 5) use optimizer to take gradient step
                optimizer.step()

    def al_step(self, train_manager):
        iterator = train_manager.get_data()
        self.train()
        self.train_loop(
            iterator,
            train_manager.criterion.to(self.device),
            train_manager.optimizer(self.parameters()),
            train_manager.num_epochs,
        )

    def get_grad_embedding(self, X, criterion, grad_embedding_type="bias_linear"):
        criterion.to(self.device)
        grad_embeddings = []

        for x in X:
            self.zero_grad()

            # calculate hypothesized labels
            if torch.is_tensor(x):
                logits = self(x.unsqueeze(dim=0))
            else:
                logits = self(np.expand_dims(x, axis=0))

            if logits.shape[1] == 1 or self.multilabel:
                logits = logits.ravel()
                y = torch.as_tensor(
                    logits > 0,
                    dtype=torch.float,
                    device=self.device,
                )
            else:
                y = torch.argmax(logits).unsqueeze(dim=0)

            loss = criterion(logits, y)
            loss.backward()

            if grad_embedding_type == "bias":
                embedding = self.out.bias.grad
            elif grad_embedding_type == "linear":
                embedding = self.out.weight.grad
            elif grad_embedding_type == "bias_linear":
                embedding = torch.cat(
                    (self.out.weight.grad, self.out.bias.grad.unsqueeze(dim=1)), dim=1
                )
            else:
                raise ValueError(
                    f"Grad embedding type '{grad_embedding_type}' not supported."
                    "Viable options: 'bias', 'linear', or 'bias_linear'"
                )
            grad_embeddings.append(embedding.flatten())

        return torch.stack(grad_embeddings)


# Order of inheritance is important for correct parent class initialization
class MLP(TorchModel, nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=32,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()

        self.output_dim = output_dim
        self.device = device

        self.hidden = nn.Linear(input_dim, hidden_dim * 2)
        self.hidden2 = nn.Linear(hidden_dim * 2, hidden_dim)
        # last layer has to be named 'out' for calculating gradient embedding
        self.out = nn.Linear(hidden_dim, output_dim)
        self.to(self.device)

    def forward(self, X):
        X = X.float()
        l1 = F.relu(self.hidden(X))
        l2 = F.relu(self.hidden2(l1))
        out = self.out(l2)
        return out

    def get_encoded(self, X):
        X = X.float()
        l1 = F.relu(self.hidden(X))
        l2 = self.hidden2(l1)
        return l2


# Order of inheritance is important for correct parent class initialization
class RNN(TorchModel, nn.Module):
    def __init__(
        self,
        output_dim,
        pretrained_embeddings,
        rnn_type,
        embedding_dim=300,
        hidden_dim=300,
        num_layers=1,
        bidirectional=True,
        dropout_p=0.5,
        padding_idx=1,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):

        super().__init__()
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.device = device

        self.embedding = nn.Embedding.from_pretrained(
            torch.from_numpy(pretrained_embeddings).float().to(self.device),
            padding_idx=padding_idx,
        )

        drop_prob = 0.0 if num_layers > 1 else dropout_p
        assert rnn_type in RNN_TYPES, f"Use one of the following: {str(RNN_TYPES)}"
        RnnCell = getattr(nn, rnn_type)
        self.rnn = RnnCell(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=drop_prob,
            batch_first=True,
        )

        self.dropout = nn.Dropout(dropout_p)
        # last layer has to be named 'out' for calculating gradient embedding
        self.out = nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim, output_dim
        )
        self.to(self.device)

    def forward(self, X):
        # X: B x S
        # print(f'X {X.shape}')

        # X = X.cuda()

        # embedded: B x S x E
        embedded = self.embedding(X)
        # print(f'embedded {embedded.shape}')

        # out: B x S x (H*num_directions)
        # hidden: B x (L*num_directions) x H
        out, hidden = self.rnn(embedded)
        # print(f'hidden {hidden.shape}')
        if type(hidden) == tuple:
            hidden = hidden[0]

        # if bidirectional concat the final forward (hidden[-1]) and
        # backward (hidden[-2]) hidden layers, otherwise extract the
        # final hidden state and apply dropout
        # hidden = B x (H*num_directions)
        if self.bidirectional:
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=-1)
        else:
            hidden = hidden[-1]

        hidden = self.dropout(hidden)
        # print(f'hidden {hidden.shape}')

        return self.out(hidden)

    def get_encoded(self, X):
        embedded = self.embedding(X)

        out, hidden = self.rnn(embedded)

        if type(hidden) == tuple:
            hidden = hidden[0]
        if self.bidirectional:
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=-1)
        else:
            hidden = hidden[-1]

        return hidden


class PackedRNN(RNN):
    def forward(self, batch):
        # X: S x B
        X, lengths = batch

        # embedded: S x B x E
        embedded = self.embedding(X)

        # pack sequence
        # output over padding tokens are zero tensors
        # hidden: (L*num_directions) x B x H
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths)
        packed_out, hidden = self.rnn(packed_embedded)
        if type(hidden) == tuple:
            hidden = hidden[0]

        # unpack sequence
        # out: S x B x (H*num_directions)
        out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_out)

        # if bidirectional concat the final forward (hidden[-2,:,:]) and
        # backward (hidden[-1,:,:]) hidden layers, otherwise extract the
        # final hidden state and apply dropout
        # hidden = B x (H*num_directions)
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        hidden = self.dropout(hidden)
        return self.out(hidden)

    def get_encoded(self, batch):
        X, lengths = batch

        embedded = self.embedding(X)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths)
        packed_out, hidden = self.rnn(packed_embedded)
        if type(hidden) == tuple:
            hidden = hidden[0]

        out, out_lengths = nn.utils.rnn.pad_packed_sequence(packed_out)
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        return hidden


# Order of inheritance is important for correct parent class initialization
class BertClassifier(TorchModel, nn.Module):
    def __init__(
        self,
        output_dim,
        bert_model_name,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super().__init__()
        self.device = device
        self.output_dim = output_dim
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.vectorizer = get_bert_vectorizer(bert_model_name, self.device)
        config = self.bert.config
        # last layer has to be named 'out' for calculating gradient embedding
        self.out = nn.Linear(config.hidden_size, output_dim)
        self.to(self.device)
        # Xavier initalization
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, X):
        X_vec = self.vectorizer(list(X))
        out = self.bert(**X_vec)
        logits = self.out(out.last_hidden_state[:, 0, :])
        return logits

    def get_encoded(self, X):
        self.to("cpu")
        X_vec = self.vectorizer(list(X)).to("cpu")
        out = self.bert(**X_vec)
        self.to(self.device)
        return out.last_hidden_state[:, 0, :]


# Order of inheritance is important for correct parent class's function use
class LogReg(LogisticRegression, ScikitModel):
    pass


# Order of inheritance is important for correct parent class's function use
class SVM(SVC, ScikitModel):
    pass


# Order of inheritance is important for correct parent class's function use
class RandomForest(RandomForestClassifier, ScikitModel):
    pass


# Order of inheritance is important for correct parent class's function use
class MOC(MultiOutputClassifier, ScikitModel):
    def predict_proba(self, X):
        """
        MultiOutputClassifier returns the list (n_outputs) of arrays (n_samples, 2),
        where each array contains the distributions of classifier for that class for each sample.
        Transform data into array (n_samples, n_outputs) with probability of sample having a certain label.
        """
        proba = super().predict_proba(X)
        prob_list = [x[:, 0].reshape(-1, 1) for x in proba]
        return np.hstack(prob_list)


MODELS = {
    "log_reg": partial(LogReg, solver="lbfgs"),
    "linear_svm": partial(SVM, kernel="linear", probability=True),
    "kernel_svm": partial(SVM, probability=True),
    "rfc": partial(RandomForest, n_estimators=100),
    "mlp": MLP,
    "rnn": partial(RNN, rnn_type="RNN"),
    "lstm": partial(RNN, rnn_type="LSTM"),
    "gru": partial(RNN, rnn_type="GRU"),
    "bert": BertClassifier,
}


PARAM_GRID = {
    "log_reg": {"C": [10 ** i for i in range(-5, 5)]},
    "svm": {"C": [10 ** i for i in range(-5, 5)]},
    "kernel_svm": {"C": [10 ** i for i in range(-5, 5)]},
    "rfc": {"n_estimators": [50, 100, 200, 500, 1000]},
}


def get_model(name, params=None, multilabel=False):
    if not params:
        params = dict()
    try:
        model = MODELS[name](**params)

        if multilabel:
            if name not in TORCH_MODELS:
                model = MOC(model)
            else:
                model.multilabel = True

        return model

    except KeyError:
        raise ValueError("Model %s not supported" % (name))
    except Exception as e:
        print(e)
