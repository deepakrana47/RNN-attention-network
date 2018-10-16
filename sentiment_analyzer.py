import theano, theano.tensor as T
import numpy as np
import pandas as pd
from util import preprocess, init_weight
from Recurrent_Unit import LSTM, GRU, SimpleRecurrentLayer
# from theano import pp

# theano.config.optimizer='fast_compile'
# theano.config.exception_verbosity='high'
# theano.config.compute_test_value = 'warn'

import matplotlib.pyplot as plt
from sklearn.utils import shuffle

def Adam(cost, params, lr=0.002, b1=0.1, b2=0.001, e=1e-8):
    updates = []
    grads = T.grad(cost, params)
    i = theano.shared(np.float32(0.))
    i_t = i + 1.
    fix1 = 1. - (1. - b1) ** i_t
    fix2 = 1. - (1. - b2) ** i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates



class SimpleRNN:
    def __init__(self, D, hidden_layer_sizes, V):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.D = D
        self.V = V

    def model(self, activation, RecurrentUnit):
        self.f = activation

        # embedding layer parameters
        we = init_weight(self.V, self.D)

        # hidden layer parameters
        self.hidden_layers = []
        Mi = self.D
        for Mo in self.hidden_layer_sizes:
            ru = RecurrentUnit(Mi, Mo, activation)
            self.hidden_layers.append(ru)
            Mi = Mo

        # attention layer parameters
        wa = init_weight(Mi, Mi)
        ba = np.zeros(Mi)
        ua = init_weight(Mi,1)

        self.Wa = theano.shared(wa)
        self.Ba = theano.shared(ba)
        self.Ua = theano.shared(ua)

        # output layer parameters
        wo = init_weight(Mi, self.O)
        bo = np.zeros(self.O)

        # shared variable
        self.We = theano.shared(we, name="Embedding weights")
        self.Wo = theano.shared(wo, name="Output weight")
        self.Bo = theano.shared(bo, name="Output Bias")
        self.params = [self.We, self.Wa, self.Ba, self.Ua, self.Wo, self.Bo]
        for ru in self.hidden_layers:
            self.params += ru.params

        # input variables
        thx = T.ivector('X')
        thy = T.ivector('Y')
        thStartPoints = T.ivector('start_points')
        thEndPoints = T.ivector('end_points')

        # embedding layer computation
        Z = self.We[thx]                                    # size = [? x D]

        # rnn layer computation
        for ru in self.hidden_layers:
            Z = ru.output(Z, thStartPoints)                 # size = [? x H]

        # attention layer computation
        u = T.tanh(Z.dot(self.Wa) + self.Ba)                # size = [? x H]
        alpha = T.nnet.softmax(u.dot(self.Ua))              # size = [? x 1]        ( [? x H].dot([H x 1]) )
        c = T.repeat(alpha, Z.shape[1], axis=1) * Z         # size = [H]            ( [? x H]*[? x H] )

        # output layer computation
        py = T.nnet.softmax(c.dot(self.Wo) + self.Bo)       # size = [O]            ( [H].dot([H x O]) )
        py_x = py[thEndPoints, :]
        prediction = T.argmax(py_x, axis=1)

        self.predict_op = theano.function(
            inputs=[thx, thStartPoints, thEndPoints],
            outputs=prediction,
            allow_input_downcast=True
        )

        return thx, thy, thStartPoints, thEndPoints, py_x, prediction

    def fit(self, X, Y, x_val, y_val, learning_rate=10e-1, activation=T.tanh, epochs=100, batches_sz=100, show_fig=False,  RecurrentUnit=None):
        self.O = len(set(Y))

        Xvalid, Yvalid = x_val, y_val

        thx, thy, thStartPoints, thEndPoints, py_x, prediction = self.model(activation, RecurrentUnit)

        py = py_x[T.arange(thy.shape[0]), thy]
        cost = -T.mean(T.where(py <= 0.0, 0, T.log(py)))
        updates = Adam(cost, params = self.params, lr = learning_rate)

        self.train_op = theano.function(
            inputs = [thx, thy, thStartPoints, thEndPoints],
            outputs = [cost, prediction],
            updates = updates,
            allow_input_downcast=True,
        )
        costs = []
        n_total = len(X)
        n_batches = int(n_total/batches_sz)

        for i in range(epochs):
            X, Y = shuffle(X, Y)
            n_correct = 0
            cost = 0
            for j in range(n_batches):

                sequenceLengths = []
                input_sequence = []
                output = []
                for k in range(j*batches_sz, (j+1)*batches_sz):
                    input_sequence += X[k]
                    output.append(Y[k])
                    sequenceLengths.append(len(X[k]))

                startPoints = np.zeros(len(input_sequence), dtype=np.int32)
                endPoints = []
                start = 0
                for length in sequenceLengths:
                    startPoints[start] = 1
                    start += length
                    endPoints.append(start-1)

                c, p = self.train_op(input_sequence, output, startPoints, endPoints)
                cost += c
                for pj, yj in zip(p, output):
                    if pj == yj:
                        n_correct += 1

                print("batch: %d/%d"%(j, n_batches), "cost:", c, "correct rate:", (float(n_correct) / n_total))

            n_correct_valid = 0
            for j in range(len(Xvalid)):
                v = Xvalid[j]
                startPoints = np.zeros(len(Xvalid[j]))
                startPoints[0] = 1
                endPoints = [len(v) - 1]
                # p, res, res1 = self.predict_op(v, startPoints, endPoints)
                p = self.predict_op(v, startPoints, endPoints)
                for pj, yj in zip(p, [Yvalid[j]]):
                    if pj == yj:
                        n_correct_valid += 1
            print(i, "validation correct rate:", float(n_correct_valid) / len(Xvalid))
            costs.append(cost)

        if show_fig:
            plt.plot(costs)
            plt.show()

    def prediction(self, X):
        Y = []
        for x in X:
            startPoints = np.zeros(len(x))
            startPoints[0] = 1
            endPoints = [len(x) - 1]
            t1 = self.predict_op(x, startPoints, endPoints)
            Y += t1.tolist()
        return Y

if __name__ == "__main__":
    train = pd.read_csv("/media/zero/41FF48D81730BD9B/kaggle/word2vec-nlp/input/labeledTrainData.tsv", header=0,delimiter='\t')
    test = pd.read_csv("/media/zero/41FF48D81730BD9B/kaggle/word2vec-nlp/input/testData.tsv", header=0, delimiter='\t')

    x_train, y_train, x_val, y_val, x_test, y_test, word2idx, idx2word = preprocess(train, test, min_word_count=100)

    # training
    rnn = SimpleRNN(D=32, hidden_layer_sizes=[64], V=len(word2idx))
    rnn.fit(x_train, y_train, x_val, y_val, learning_rate=0.0001, activation=T.nnet.relu, show_fig=False, epochs=10, batches_sz=100, RecurrentUnit=SimpleRecurrentLayer)
    y = rnn.prediction(x_test)

    total = len(x_test)
    correct = 0.0
    for i in range(len(y)):
        if y[i] == y_test[i]:
            correct += 1
    print "accuracy : ", correct / total
    # Vannila RNN, stopword removed, attention weight and attention vector [epochs=5, learning_rate=0.0001, word_embedding=32, hiddenlayer=[64], batch_size=100, op_method=adam, min_word_count=100]
    # ('batch: 199/200', 'cost:', array(0.3859546), 'correct rate:', 0.8677)
    # (4, 'validation correct rate:', 0.8406)
    # accuracy :  0.82684

    # Vannila RNN, stopword removed, attention weight and attention vector [epochs=7, learning_rate=0.0001, word_embedding=32, hiddenlayer=[64], batch_size=100, op_method=adam, min_word_count=100]
    # ('batch: 199/200', 'cost:', array(0.29866783), 'correct rate:', 0.89575)
    # (6, 'validation correct rate:', 0.8706)
    # accuracy :  0.85976

    # Vannila RNN, stopword removed, attention weight and attention vector [epochs=10, learning_rate=0.0001, word_embedding=32, hiddenlayer=[64], batch_size=100, op_method=adam, min_word_count=100]
    # ('batch: 199/200', 'cost:', array(0.26357018), 'correct rate:', 0.9235)
    # (9, 'validation correct rate:', 0.8692)
    # accuracy :  0.86016

    # GRU, attention vector [epochs=10, word_embedding=32, learning_rate=0.002, hiddenlayer=[64], batch_size=100, op_method=adam, num_most_freq_words_to_include = 500]
    # ('validation correct rate:', 0.847)
    # accuracy:  0.84512

    # GRU, attention weight and attention vector [epochs=10, learning_rate=0.002, word_embedding=32, hiddenlayer=[64], batch_size=100, op_method=adam, num_most_freq_words_to_include = 500]
    # ('validation correct rate:', 0.8534)
    # accuracy:  0.84656

    # GRU, stopword removed, attention weight and attention vector [epochs=5, learning_rate=0.0001, word_embedding=32, hiddenlayer=[64], batch_size=100, op_method=adam, num_most_freq_words_to_include = 5000]
    # ('4 validation correct rate:', (0.8602))
    # accuracy :  0.85468

    # GRU, stopword removed, attention weight and attention vector [epochs=10, learning_rate=0.0001, word_embedding=32, hiddenlayer=[64], batch_size=100, op_method=adam, num_most_freq_words_to_include = 5000]
    # ('validation correct rate:', 0.858)
    # accuracy :  0.84416

    # GRU, stopword removed, attention weight and attention vector [epochs=5, learning_rate=0.0001, word_embedding=32, hiddenlayer=[64], batch_size=100, op_method=adam, num_most_freq_words_to_include = 10000]
    # ('batch: 199/200', 'cost:', array(0.37813238), 'correct rate:', 0.90265
    # (4, 'validation correct rate:', 0.87)
    # accuracy :  0.85176

    # GRU, stopword removed, attention weight and attention vector [epochs=5, learning_rate=0.0001, word_embedding=32, hiddenlayer=[64], batch_size=100, op_method=adam, min_word_count=40]
    # ('batch: 199/200', 'cost:', array(0.3827904), 'correct rate:', 0.8486)
    # (4, 'validation correct rate:', 0.8588)
    # accuracy :  0.84384

    # GRU, stopword removed, attention weight and attention vector [epochs=5, learning_rate=0.0001, word_embedding=32, hiddenlayer=[64], batch_size=100, op_method=adam, min_word_count=100]
    # ('batch: 199/200', 'cost:', array(0.41816997), 'correct rate:', 0.88855)
    # (4, 'validation correct rate:', 0.865)
    # accuracy :  0.84888

    # LSTM stopword removed, attention weight and attention vector [epochs=5, learning_rate=0.0001, word_embedding=32, hiddenlayer=[64], batch_size=100, op_method=adam, min_word_count=100]
    # ('batch: 199/200', 'cost:', array(0.56432788), 'correct rate:', 0.84755)
    # (4, 'validation correct rate:', 0.8096)
    # accuracy :  0.8078

    # LSTM stopword removed, attention weight and attention vector [epochs=7, learning_rate=0.0001, word_embedding=32, hiddenlayer=[64], batch_size=100, op_method=adam, min_word_count=100]
    # ('batch: 199/200', 'cost:', array(0.45407547), 'correct rate:', 0.8668)
    # (6, 'validation correct rate:', 0.8616)
    # accuracy :  0.84108

