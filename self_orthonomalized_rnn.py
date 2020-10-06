from collections import OrderedDict
import copy
import gzip
import pickle
import json
import time
import numpy as np
from matplotlib import pyplot as plt
import theano
from theano import tensor as T
from numpy.random import RandomState as NP_RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as TH_RandomStreams
from theano.ifelse import ifelse

RAND_SEED = 123456798

# mode = 'DebugMode'
# mode = theano.Mode(linker='cvm', optimizer='fast_compile')
mode = theano.Mode(linker='cvm', optimizer='fast_run')
# mode = theano.Mode(linker='cvm_nogc', optimizer='fast_run')
profile = False

class TheanoOrthonomalizedAdamParams:
    def __init__(self, param_info, dtype, borrow, np_rng, clip_grad=-1,
                 decoupling_list=(), orth_init_list=()):
        self.borrow = borrow
        self.clip_grad = clip_grad
        self.param_info = param_info  # OrderDict([name, shape])
        self.decoupling_list = decoupling_list
        self.params, self.mg_params, self.vg_params = [], [], []
        for name, shape in param_info.items():
            if name[0] == 'b':  # bias
                value_init = np.zeros(shape)
            elif name in orth_init_list:
                u, _, vh = np.linalg.svd(np_rng.randn(*shape),
                                         full_matrices=False)
                value_init = np.dot(u, vh)
            else:  # Xavier 2010
                rang = np.sqrt(6/(sum(shape)))
                value_init = np_rng.uniform(size=shape, low=-rang, high=rang)

            # Replace the shared variables in decoupling_list by decoupled
            # tensor variables. Since Theano cant not back prop efficiently
            # through self-orthonomalization, thus we need to do it manually.
            # Speed up factor: 50x
            if name in decoupling_list:
                self.__dict__['predcpl_'+name] = theano.shared(
                    value=value_init.astype(dtype),
                    name='predcpl_'+name, borrow=borrow
                )
                self.__dict__[name] = T.matrix(name=name)
            else:
                self.__dict__[name] = theano.shared(
                    value=value_init.astype(dtype),
                    name=name, borrow=borrow
                )
            self.params.append(self.__dict__[name])

            self.__dict__['mg_'+name] = theano.shared(
                value=np.zeros_like(value_init).astype(dtype),
                name='mg_'+name, borrow=borrow
            )  # Adam's cache of grad
            self.__dict__['vg_'+name] = theano.shared(
                value=np.zeros_like(value_init).astype(dtype),
                name='vg_'+name, borrow=borrow
            )  # Adam's cache of grad**2

        # Iteration
        self.it = theano.shared(
            value=np.array(1., dtype=dtype), name='it', borrow=borrow
        )

    def get_value(self):
        value = OrderedDict([])
        for name, var in self.__dict__.items():
            if isinstance(var, theano.compile.sharedvalue.SharedVariable):
                value.update({name: var.get_value(borrow=self.borrow)})
        return value

    def set_value(self, value):
        for name, val in value.items():
            self.__dict__[name].set_value(val, borrow=self.borrow)

    def get_adam_updates(self, cost, lr, b1, b2, eps):
        grads = T.grad(cost, self.params)
        if self.clip_grad > 0:
            grads = T.clip(grads, -self.clip_grad, self.clip_grad)
        lr_iter = lr * T.sqrt(1 - b2**self.it)/(1 - b1**self.it)

        givens = OrderedDict({})
        updates = OrderedDict({self.it: self.it+1})
        for i, name in enumerate(self.param_info.keys()):
            if name in self.decoupling_list:
                param = self.__dict__['predcpl_'+name]
                # orthogonalization
                u_p, s_p, vt_p = T.nlinalg.svd(param)
                givens.update({self.__dict__[name]: T.dot(u_p, vt_p)})
                # Manually back-propagate through orthogonalization, 50x speed
                # improvement
                c_p = -T.nlinalg.matrix_dot(
                            u_p.T/s_p[:, None], grads[i], vt_p.T
                        ) / (s_p[:, None]+s_p[None, :])
                # Grad wrt pre-orthogonalization parameter
                grad = T.nlinalg.matrix_dot(
                            u_p, c_p+c_p.T, vt_p*s_p[:, None]
                        ) + T.nlinalg.matrix_dot(u_p/s_p, u_p.T, grads[i])
            else:
                param = self.__dict__[name]
                grad = grads[i]

            mg_param = self.__dict__['mg_'+name]
            vg_param = self.__dict__['vg_'+name]

            # Adam updating rules
            mg_param_new = b1*mg_param + (1-b1) * grad
            vg_param_new = b2*vg_param + (1-b2) * grad**2
            param_new = param - \
                        lr_iter*mg_param_new/(T.sqrt(vg_param_new) + eps)
            updates.update({param: param_new,
                            mg_param: mg_param_new,
                            vg_param: vg_param_new})

        return givens, updates


class BaseRNN:
    def __init__(self, data_dim, hidden_dim, out_dim, output_type,
                 activation=T.tanh, dtype=theano.config.floatX, borrow=False,
                 decoupling=False, truncate_gradient=-1, lmbda=0., dropout=0.,
                 np_rng=NP_RandomStreams(RAND_SEED),
                 th_rng=TH_RandomStreams(RAND_SEED),
                 allow_input_downcast=True):
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.activation = activation
        self.dtype = dtype
        self.borrow = borrow
        self.decoupling = decoupling
        self.output_type = output_type
        self.truncate_gradient = truncate_gradient
        self.np_rng = np_rng
        self.th_rng = th_rng
        self.lmbda = lmbda  # L2-regularized coefficient of output weights
        self.dropout = dropout

        # Create symbolic variable for input and output
        #    x          mask
        # 1 1 1 1 ..    # 1 1 1 1 ..
        # 0 2 2 2 ..    # 0 1 1 1 ..
        # 0 0 3 3 ..    # 0 0 1 1 ..
        self.x = T.lmatrix('x')      # x.shape: padded_seq_len, n_samples
        self.padded_seq_len, self.n_samples = self.x.shape
        self.mask = T.matrix('mask')
        if self.output_type == 'sequence':
            self.y = T.lmatrix('y')  # y.shape: padded_seq_len, n_samples
        elif self.output_type == 'discrete':
            self.y = T.lvector('y')  # y.shape: n_samples,
        elif self.output_type == 'real':
            self.y = T.vector('y')   # y.shape: n_samples,
        else:
            raise NotImplementedError

        # We need this variable to turn off drop out in prediction
        self.is_training = T.iscalar('is_training')

        self.params = self.allocate_and_init_parameters()
        h = self.recurrence(self.params)
        self.cost, self.pred_error, self.pred, self.pred_prob = \
            self.build_output_layer(self.params, h)

        self.lr = T.scalar('lr')
        self.b1 = T.scalar('b1')
        self.b2 = T.scalar('b2')
        self.eps = T.scalar('eps')

        # Find the gradient of pre-orthogonalization model and build Adam update
        givens, updates = self.params.get_adam_updates(
            self.cost, self.lr, self.b1, self.b2, self.eps
        )
        training_givens = OrderedDict(givens, **{self.is_training: 1})
        prediction_givens = OrderedDict(givens, **{self.is_training: 0})

        # Compile theano functions
        self.train_on_batch = theano.function(
            inputs=[self.x, self.mask, self.y,
                    self.lr, self.b1, self.b2, self.eps],
            outputs=self.cost,
            givens=training_givens,
            updates=updates, profile=profile,
            mode=mode, allow_input_downcast=allow_input_downcast
        )

        self.loss = theano.function(
            inputs=[self.x, self.mask, self.y], outputs=self.cost,
            givens=prediction_givens,
            profile=profile, mode=mode,
            allow_input_downcast=allow_input_downcast
        )

        # Zero_one loss
        self.error = theano.function(
            inputs=[self.x, self.mask, self.y], outputs=self.pred_error,
            givens=prediction_givens,
            profile=profile, mode=mode,
            allow_input_downcast=allow_input_downcast
        )

        self.predict_prob = theano.function(
            inputs=[self.x, self.mask], outputs=self.pred_prob,
            givens=prediction_givens,
            profile=profile, mode=mode,
            allow_input_downcast=allow_input_downcast
        )

        self.predict = theano.function(
            inputs=[self.x, self.mask], outputs=self.pred,
            givens=prediction_givens,
            profile=profile, mode=mode,
            allow_input_downcast=allow_input_downcast
        )

    def allocate_and_init_parameters(self):
        """Allocate and initialize all parameters

        Numpy'array default layout is row, so instead of W.dot(x)
        we use x.T.dot(W.T).
        :return: an instance of TheanoAdamParams
        """
        param_info = OrderedDict([
            ('W_xe', (self.data_dim, self.hidden_dim)),    # Embedding matrix
            ('W_hh', (self.hidden_dim, self.hidden_dim)),  # state transition
            ('W_eh', (self.hidden_dim, self.hidden_dim)),
            ('b_h', (self.hidden_dim,)),  # bias symbols must start with b
            ('W_ho', (self.hidden_dim, out_dim)),
            ('b_o', (self.out_dim,)),
        ])
        orth_init_list = ['W_hh', 'W_eh']
        if self.decoupling:
            decoupling_list = ['W_hh', 'W_eh']
            clip_grad = -1
        else:
            decoupling_list = []
            clip_grad = 1

        return TheanoOrthonomalizedAdamParams(
            param_info, self.dtype, self.borrow,
            self.np_rng, clip_grad, decoupling_list, orth_init_list
        )

    def recurrence(self, p):
        # x.shape = [padded_seq_len, n_samples]
        # one hot encoding of x: [padded_seq_len, n_samples, data_dim]
        # We compute the embedding, i.e., one_hot_x.dot(W_xe) efficiently by
        e = p.W_xe[self.x.flatten()].reshape(
            [self.padded_seq_len, self.n_samples, self.hidden_dim]
        )

        # recurrent function
        def step(m_t, e_t, h_tm1):
            h_t = self.activation(T.dot(h_tm1, p.W_hh) +
                                  T.dot(e_t, p.W_eh) + p.b_h)
            #Stop updating h_t after the end of the sequence, not necessary
            h_t = m_t[:, None] * h_t + (1-m_t)[:, None] * h_tm1
            return h_t

        # h.shape: padded_seq_len, n_samples, hidden_dim
        h, updates = theano.scan(
            fn=step, sequences=[self.mask, e],
            outputs_info=[
                T.zeros([self.n_samples, self.hidden_dim], dtype=self.dtype)
            ],
            n_steps=self.padded_seq_len,
            truncate_gradient=self.truncate_gradient
        )

        return h

    def build_output_layer(self, p, h):
        scaling_mask = self.mask / self.mask.sum(axis=0, keepdims=True)
        # Find average hidden state if we are performing classication or
        # regression.
        if self.output_type != 'sequence':
            h = T.sum(h*scaling_mask[:, :, None], axis=0)

        # Drop out
        if 0 < self.dropout <1:
            h = ifelse(
                self.is_training,
                h * self.th_rng.binomial(
                    h.shape, p=self.dropout, n=1, dtype=self.dtype
                ) / self.dropout,
                h
            )

        # o.shape: [padded_seq_len, n_samples, out_dim] for sequence output
        # o.shape: [n_samples, out_dim] for other output_type
        o = (T.dot(h, p.W_ho) + p.b_o)

        if self.output_type == 'sequence':
            # Theano nnet functions work on 2D array only, so we need to
            # flatten the first 2 dims
            o = o.reshape([self.padded_seq_len*self.n_samples, -1])
            pred_prob2d = T.nnet.softmax(o)
            cross_ent = T.nnet.categorical_crossentropy(
                pred_prob2d, self.y.flatten()
            ).reshape([self.padded_seq_len, self.n_samples] )

            # Clear the output which correspond to padded symbols
            # and divided cross-entropy by sequence lengths => avg x-entropy
            cost = T.sum(cross_ent*scaling_mask) / self.n_samples

            pred_prob = pred_prob2d.reshape(
                [self.padded_seq_len, self.n_samples, self.out_dim]
            )
            # padded_seq_len, n_samples
            pred = T.argmax(pred_prob, axis=2)
            pred_error = T.sum(T.neq(pred, self.y)*scaling_mask)/self.n_samples
        elif self.output_type == 'discrete':
            # Clear the output which correspond to padded symbols,
            # and divided it by sequence lengths, sum across time => mean output
            pred_prob = T.nnet.softmax(o)
            cross_ent = T.nnet.categorical_crossentropy(pred_prob, self.y)
            cost = T.mean(cross_ent)
            # pred.shape: n_samples,
            pred = T.argmax(pred_prob, axis=1)
            pred_error = T.mean(T.neq(pred, self.y))
        elif self.output_type == 'real':
            pred = o
            # Find the average hidden state of each sequence
            # o.shape is [n_samples, 1]
            cost = T.mean((pred - self.y)**2)  # MSE error
            pred_prob = pred  # Non-sense
            pred_error = T.mean((pred - self.y)**2)
        else:
            raise NotImplementedError

        if self.lmbda > 0:
            lmbda_ = theano.shared(value=np.array(self.lmbda, dtype=self.dtype),
                                   name='lambda', borrow=self.borrow)
            # L2 regularization
            cost += lmbda_ * (T.sum((p.W_ho+p.W_ho)**2))
        # Note: Prediction error maybe abit misleading for the last batch

        return cost, pred_error, pred, pred_prob

    def fit(self, X, Y,
            learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8,
            valid_percent=0.1, valid_batch_size=128, patience=10,
            valid_freq=500, batch_size=16, max_epoch=1000, max_len=None):
        # Filter out long sequences
        if max_len is not None:
            X[:], Y[:] = zip(*[(x, y) for (x, y) in zip(X, Y)
                               if len(x) <= max_len])

        # Divide the dataset into training set and validation set
        n_total_samples = len(X)
        n_train = int(n_total_samples * (1 - valid_percent))
        n_valid = n_total_samples - n_train
        perm_idx = self.np_rng.permutation(n_total_samples)
        X_train, X_valid = X[perm_idx[:n_train]], X[perm_idx[n_train:]]
        Y_train, Y_valid = Y[perm_idx[:n_train]], Y[perm_idx[n_train:]]
        n_batches = int(np.ceil(len(X_train) / batch_size))

        it = 0  # Number of update or iteration
        best_params = None
        cost_hist = []
        ve_hist = []  # validation error history
        te_hist = []  # training error history
        bad_counter = 0
        best_vc = np.Inf
        try:
            for epoch in range(max_epoch):
                b_idx = 0
                for X_bat, Y_bat in self.chunk(X_train, Y_train,
                                               batch_size, self.np_rng):
                    it += 1
                    b_idx += 1

                    # Training
                    x, mask, y = self.zeropad(X_bat, Y_bat)
                    start_time = time.time()
                    c = self.train_on_batch(x, mask, y, learning_rate,
                                            beta1, beta2, epsilon)
                    elapse_time = time.time() - start_time
                    if np.isnan(c) or np.isinf(c):
                        print(
                            'Bad cost={0} at {}th step!'.format(c, it)
                        )
                        raise ValueError
                    cost_hist.append((it, c))

                    # Training error for current batch
                    te = self.error(x, mask, y)
                    te_hist.append((it, te))

                    # Validation at the initial value too.
                    # Set valid_freq to zero if we do not use early stopping
                    if valid_freq and np.mod(it-1, valid_freq) == 0:
                        ve = 0  # validation error
                        for Xv_bat, Yv_bat in self.chunk(X_valid, Y_valid,
                                                         valid_batch_size):
                            x_vld, m_vld, y_vld = self.zeropad(Xv_bat, Yv_bat)
                            ve += self.error(x_vld, m_vld, y_vld)*x_vld.shape[1]
                        ve /= n_valid
                        ve_hist.append((it, ve))

                        print('Epoch={:3d}: batch={:5d}/{}, update={:6d}, '
                              'minibatch_cost={:6.5f}, time/batch={:.4f}s, '
                              'training_error={:6.5f}, '
                              'valid_error={:6.5f}'.format(
                            epoch, b_idx, n_batches, it, float(c),
                            elapse_time, float(te), float(ve)
                        ))

                        # Early stopping
                        if ve <= best_vc:
                            best_params = copy.deepcopy(self.params.get_value())
                            best_vc = ve
                            bad_counter = 0
                        elif len(ve_hist) > patience:
                            if ve > np.asarray(ve_hist)[:-patience, 1].min():
                                bad_counter += 1
                                if bad_counter > patience:
                                    print('Early stop!')
                                    raise StopIteration



        except (KeyboardInterrupt, StopIteration, ValueError) as e:
            print('Stop training!')

        if best_params is not None:
            self.params.set_value(best_params)

        cost_hist = np.asarray(cost_hist)
        ve_hist = np.asarray(ve_hist)
        te_hist = np.asarray(te_hist)
        print('Best validation error='+str(ve_hist[:, 1].min()))
        return cost_hist, ve_hist, te_hist

    def chunk(self, X, Y, batch_size, np_rng=None):
        n_total_samples = len(X)
        if np_rng is not None:
            # Shuffle the samples if np_rng is not None
            idx = np_rng.permutation(n_total_samples)
            for i in range(0, n_total_samples, batch_size):
                yield X[idx[i:i + batch_size]], Y[idx[i:i + batch_size]]
        else:
            for i in range(0, n_total_samples, batch_size):
                yield X[i:i + batch_size], Y[i:i + batch_size]

    # Padding and convert to np.array
    def zeropad(self, seq_in, seq_out=None):
        n_total_samples = len(seq_in)
        seq_len = [len(s) for s in seq_in]
        max_len = np.max(seq_len)

        padded_seq_in = np.zeros((max_len, n_total_samples))
        seq_in_mask = np.zeros((max_len, n_total_samples))
        for n in range(n_total_samples):
            padded_seq_in[:seq_len[n], n] = seq_in[n]
            seq_in_mask[:seq_len[n], n] = 1

        padded_seq_out = None
        if seq_out is not None:
            if self.output_type != 'sequence':
                padded_seq_out = np.array(seq_out)
            else:
                # if output are also sequence
                padded_seq_out = np.zeros((max_len, n_total_samples))
                for n in range(n_total_samples):
                    padded_seq_out[:seq_len[n], n] = seq_out[n]

        return padded_seq_in, seq_in_mask, padded_seq_out


class RNN2layers(BaseRNN):
    def __init__(self, data_dim, hidden_dim, out_dim, output_type,
                 activation=T.tanh, dtype=theano.config.floatX, borrow=False,
                 decoupling=False, truncate_gradient=-1, lmbda=0., dropout=0.,
                 np_rng=NP_RandomStreams(RAND_SEED),
                 th_rng=TH_RandomStreams(RAND_SEED),
                 allow_input_downcast=True):
        super().__init__(data_dim, hidden_dim, out_dim, output_type,
                 activation, dtype, borrow, decoupling, truncate_gradient,
                 lmbda, dropout, np_rng, th_rng, allow_input_downcast)

    def allocate_and_init_parameters(self):
        """Allocate and initialize all parameters

        Numpy'array default layout is row, so instead of W.dot(x)
        we use x.T.dot(W.T).
        :return: an instance of TheanoAdamParams
        """
        param_info = OrderedDict([
            ('W_xe', (self.data_dim, self.hidden_dim)),    # Embedding matrix
            ('W_hh0', (self.hidden_dim, self.hidden_dim)),  # state transition
            ('W_eh0', (self.hidden_dim, self.hidden_dim)),
            ('b_h0', (self.hidden_dim,)),  # bias symbols must start with b
            ('W_hh1', (self.hidden_dim, self.hidden_dim)),  # state transition
            ('W_eh1', (self.hidden_dim, self.hidden_dim)),
            ('b_h1', (self.hidden_dim,)),  # bias symbols must start with b
            ('W_ho', (self.hidden_dim, out_dim)),
            ('b_o', (self.out_dim,)),
        ])
        orth_init_list = ['W_hh0', 'W_eh0', 'W_hh1', 'W_eh1']
        if self.decoupling:
            decoupling_list = ['W_hh0', 'W_hh1']
            clip_grad = -1
        else:
            decoupling_list = []
            clip_grad = 1

        return TheanoOrthonomalizedAdamParams(
            param_info, self.dtype, self.borrow,
            self.np_rng, clip_grad, decoupling_list, orth_init_list
        )

    def recurrence(self, p):
        # x.shape = [padded_seq_len, n_samples]
        # one hot encoding of x: [padded_seq_len, n_samples, data_dim]
        # We compute the embedding, i.e., one_hot_x.dot(W_xe) efficiently by
        e = p.W_xe[self.x.flatten()].reshape(
            [self.padded_seq_len, self.n_samples, self.hidden_dim]
        )

        # recurrent function
        def step(m_t, e_t, h0_tm1, h1_tm1):
            h0_t = self.activation(T.dot(h0_tm1, p.W_hh0) +
                                  T.dot(e_t, p.W_eh0) + p.b_h0)
            h1_t = self.activation(T.dot(h1_tm1, p.W_hh1) +
                                  T.dot(h0_t, p.W_eh1) + p.b_h1)
            #Stop updating h_t after the end of the sequence, not necessary
            h0_t = m_t[:, None] * h0_t + (1-m_t)[:, None] * h0_tm1
            h1_t = m_t[:, None] * h1_t + (1-m_t)[:, None] * h1_tm1
            return h0_t, h1_t

        # h.shape: padded_seq_len, n_samples, hidden_dim
        [_, h], updates = theano.scan(
            fn=step, sequences=[self.mask, e],
            outputs_info=[
                T.zeros([self.n_samples, self.hidden_dim], dtype=self.dtype),
                T.zeros([self.n_samples, self.hidden_dim], dtype=self.dtype)
            ],
            n_steps=self.padded_seq_len,
            truncate_gradient=self.truncate_gradient
        )

        return h


class UGRNN(BaseRNN):
    def __init__(self, data_dim, hidden_dim, out_dim, output_type,
                 activation=T.tanh, dtype=theano.config.floatX, borrow=False,
                 decoupling=False, truncate_gradient=-1, lmbda=0., dropout=0.,
                 np_rng=NP_RandomStreams(RAND_SEED),
                 th_rng=TH_RandomStreams(RAND_SEED),
                 allow_input_downcast=True):
        super().__init__(data_dim, hidden_dim, out_dim, output_type,
                 activation, dtype, borrow, decoupling, truncate_gradient,
                 lmbda, dropout, np_rng, th_rng, allow_input_downcast)

    def allocate_and_init_parameters(self):
        """Allocate and initialize all parameters

        Numpy'array default layout is row, so instead of W.dot(x)
        we use x.T.dot(W.T).
        :return: an instance of TheanoAdamParams
        """
        # bias symbols must start with b
        param_info = OrderedDict([
            ('W_xe', (self.data_dim, self.hidden_dim)),    # Embedding matrix
            ('W_hu', (self.hidden_dim, self.hidden_dim)),  # update gate
            ('W_eu', (self.hidden_dim, self.hidden_dim)),
            ('b_u', (self.hidden_dim,)),
            ('W_hc', (self.hidden_dim, self.hidden_dim)),  # candidate update
            ('W_ec', (self.hidden_dim, self.hidden_dim)),
            ('b_c', (self.hidden_dim,)),  #
            ('W_ho', (self.hidden_dim, out_dim)),
            ('b_o', (self.out_dim,)),
        ])
        orth_init_list = ['W_hu', 'W_hc', 'W_eu', 'W_ec']
        if self.decoupling:
            decoupling_list = ['W_hu', 'W_hc', 'W_eu', 'W_ec']
            clip_grad = -1
        else:
            decoupling_list = []
            clip_grad = 1

        return TheanoOrthonomalizedAdamParams(
            param_info, self.dtype, self.borrow,
            self.np_rng, clip_grad, decoupling_list, orth_init_list
        )

    def recurrence(self, p):
        # x.shape = [padded_seq_len, n_samples]
        # one hot encoding of x: [padded_seq_len, n_samples, data_dim]
        # We compute the embedding, i.e., one_hot_x.dot(W_xe) efficiently by
        e = p.W_xe[self.x.flatten()].reshape(
            [self.padded_seq_len, self.n_samples, self.hidden_dim]
        )

        # recurrent function
        def step(m_t, e_t, h_tm1):
            u = T.nnet.sigmoid(T.dot(e_t, p.W_eu) +
                               T.dot(h_tm1, p.W_hu) + p.b_u)
            c = self.activation(T.dot(e_t, p.W_ec) +
                                T.dot(h_tm1, p.W_hc) + p.b_c)
            h_t = (1-u)*h_tm1 + u*c
            #Stop updating h_t after the end of the sequence, not necessary
            h_t = m_t[:, None] * h_t + (1-m_t)[:, None] * h_tm1
            return h_t

        # h.shape: padded_seq_len, n_samples, hidden_dim
        h, updates = theano.scan(
            fn=step, sequences=[self.mask, e],
            outputs_info=[
                T.zeros([self.n_samples, self.hidden_dim], dtype=self.dtype)
            ],
            n_steps=self.padded_seq_len,
            truncate_gradient=self.truncate_gradient
        )

        return h


class GRU(BaseRNN):
    def __init__(self, data_dim, hidden_dim, out_dim, output_type,
                 activation=T.tanh, dtype=theano.config.floatX, borrow=False,
                 decoupling=False, truncate_gradient=-1, lmbda=0., dropout=0.,
                 np_rng=NP_RandomStreams(RAND_SEED),
                 th_rng=TH_RandomStreams(RAND_SEED),
                 allow_input_downcast=True):
        super().__init__(data_dim, hidden_dim, out_dim, output_type,
                 activation, dtype, borrow, decoupling, truncate_gradient,
                 lmbda, dropout, np_rng, th_rng, allow_input_downcast)

    def allocate_and_init_parameters(self):
        """Allocate and initialize all parameters

        Numpy'array default layout is row, so instead of W.dot(x)
        we use x.T.dot(W.T).
        :return: an instance of TheanoAdamParams
        """
        # bias symbols must start with b
        param_info = OrderedDict([
            ('W_xe', (self.data_dim, self.hidden_dim)),    # Embedding matrix
            ('W_hr', (self.hidden_dim, self.hidden_dim)),  # reset gate
            ('W_er', (self.hidden_dim, self.hidden_dim)),
            ('b_r',  (self.hidden_dim,)),
            ('W_hu', (self.hidden_dim, self.hidden_dim)),  # update gate
            ('W_eu', (self.hidden_dim, self.hidden_dim)),
            ('b_u', (self.hidden_dim,)),
            ('W_hc', (self.hidden_dim, self.hidden_dim)),  # candidate update
            ('W_ec', (self.hidden_dim, self.hidden_dim)),
            ('b_c', (self.hidden_dim,)),  #
            ('W_ho', (self.hidden_dim, out_dim)),
            ('b_o', (self.out_dim,)),
        ])
        orth_init_list = ['W_hr', 'W_hu', 'W_hc', 'W_er', 'W_eu', 'W_ec']
        if self.decoupling:
            decoupling_list = ['W_hr', 'W_hu', 'W_hc', 'W_er', 'W_eu', 'W_ec']
            clip_grad = -1
        else:
            decoupling_list = []
            clip_grad = 1

        return TheanoOrthonomalizedAdamParams(
            param_info, self.dtype, self.borrow,
            self.np_rng, clip_grad, decoupling_list, orth_init_list
        )

    def recurrence(self, p):
        # x.shape = [padded_seq_len, n_samples]
        # one hot encoding of x: [padded_seq_len, n_samples, data_dim]
        # We compute the embedding, i.e., one_hot_x.dot(W_xe) efficiently by
        e = p.W_xe[self.x.flatten()].reshape(
            [self.padded_seq_len, self.n_samples, self.hidden_dim]
        )

        # recurrent function
        def step(m_t, e_t, h_tm1):
            r = T.nnet.sigmoid(T.dot(e_t, p.W_er) +
                               T.dot(h_tm1, p.W_hr) + p.b_r)
            u = T.nnet.sigmoid(T.dot(e_t, p.W_eu) +
                               T.dot(h_tm1, p.W_hu) + p.b_u)
            c = self.activation(T.dot(e_t, p.W_ec) +
                                T.dot(r*h_tm1, p.W_hc) + p.b_c)
            h_t = (1-u)*h_tm1 + u*c
            #Stop updating h_t after the end of the sequence, not necessary
            h_t = m_t[:, None] * h_t + (1-m_t)[:, None] * h_tm1
            return h_t

        # h.shape: padded_seq_len, n_samples, hidden_dim
        h, updates = theano.scan(
            fn=step, sequences=[self.mask, e],
            outputs_info=[
                T.zeros([self.n_samples, self.hidden_dim], dtype=self.dtype)
            ],
            n_steps=self.padded_seq_len,
            truncate_gradient=self.truncate_gradient
        )

        return h


class LSTM(BaseRNN):
    def __init__(self, data_dim, hidden_dim, out_dim, output_type,
                 activation=T.tanh, dtype=theano.config.floatX, borrow=False,
                 decoupling=False, truncate_gradient=-1, lmbda=0., dropout=0.,
                 np_rng=NP_RandomStreams(RAND_SEED),
                 th_rng=TH_RandomStreams(RAND_SEED),
                 allow_input_downcast=True):
        super().__init__(data_dim, hidden_dim, out_dim, output_type,
                 activation, dtype, borrow, decoupling, truncate_gradient,
                 lmbda, dropout, np_rng, th_rng, allow_input_downcast)

    def allocate_and_init_parameters(self):
        """Allocate and initialize all parameters

        Numpy'array default layout is row, so instead of W.dot(x)
        we use x.T.dot(W.T).
        :return: an instance of TheanoAdamParams
        """
        # bias symbols must start with b
        param_info = OrderedDict([
            ('W_xe', (self.data_dim, self.hidden_dim)),    # Embedding matrix
            ('W_hi', (self.hidden_dim, self.hidden_dim)),  # input gate
            ('W_ei', (self.hidden_dim, self.hidden_dim)),
            ('b_i',  (self.hidden_dim,)),
            ('W_hf', (self.hidden_dim, self.hidden_dim)),  # forget gate
            ('W_ef', (self.hidden_dim, self.hidden_dim)),
            ('b_f', (self.hidden_dim,)),
            ('W_hc', (self.hidden_dim, self.hidden_dim)),  # candidate update
            ('W_ec', (self.hidden_dim, self.hidden_dim)),
            ('b_c', (self.hidden_dim,)),
            ('W_hz', (self.hidden_dim, self.hidden_dim)),  # output gate
            ('W_ez', (self.hidden_dim, self.hidden_dim)),
            ('b_z', (self.hidden_dim,)),
            ('W_ho', (self.hidden_dim, out_dim)),
            ('b_o', (self.out_dim,)),
        ])
        orth_init_list = ['W_hi', 'W_hf', 'W_hc', 'W_hz',
                          'W_ei', 'W_ef', 'W_ec', 'W_ez']
        if self.decoupling:
            decoupling_list = ['W_hi', 'W_hf', 'W_hc', 'W_hz',
                               'W_ei', 'W_ef', 'W_ec', 'W_ez']
            clip_grad = -1
        else:
            decoupling_list = []
            clip_grad = 1

        return TheanoOrthonomalizedAdamParams(
            param_info, self.dtype, self.borrow,
            self.np_rng, clip_grad, decoupling_list, orth_init_list
        )

    def recurrence(self, p):
        # x.shape = [padded_seq_len, n_samples]
        # one hot encoding of x: [padded_seq_len, n_samples, data_dim]
        # We compute the embedding, i.e., one_hot_x.dot(W_xe) efficiently by
        e = p.W_xe[self.x.flatten()].reshape(
            [self.padded_seq_len, self.n_samples, self.hidden_dim]
        )

        # recurrent function
        def step(m_t, e_t, h_tm1, c_tm1):
            i = T.nnet.sigmoid(T.dot(e_t, p.W_ei) +
                               T.dot(h_tm1, p.W_hi) + p.b_i)
            f = T.nnet.sigmoid(T.dot(e_t, p.W_ef) +
                               T.dot(h_tm1, p.W_hf) + p.b_f)
            c_ = self.activation(T.dot(e_t, p.W_ec) +
                                 T.dot(h_tm1, p.W_hc) + p.b_c)
            c_t = f*c_tm1 + i*c_

            z = T.nnet.sigmoid(T.dot(e_t, p.W_ez) +
                               T.dot(h_tm1, p.W_hz) + p.b_z)
            h_t = z*T.tanh(c_t)

            #Stop updating h_t after the end of the sequence, not necessary
            c_t = m_t[:, None] * c_t + (1-m_t)[:, None] * c_tm1
            h_t = m_t[:, None] * h_t + (1-m_t)[:, None] * h_tm1
            return h_t, c_t

        # h.shape: padded_seq_len, n_samples, hidden_dim
        [h, _], updates = theano.scan(
            fn=step, sequences=[self.mask, e],
            outputs_info=[
                T.zeros([self.n_samples, self.hidden_dim], dtype=self.dtype),
                T.zeros([self.n_samples, self.hidden_dim], dtype=self.dtype),
            ],
            n_steps=self.padded_seq_len,
            truncate_gradient=self.truncate_gradient
        )

        return h


# TODO: Implement mnist, memorization, etc



if __name__ == '__main__':
    with open('processed_science_titles.json', mode='r', encoding='utf-8') as f:
        dataset = json.load(f)

    i2w = dataset['i2w']
    word_dim = len(i2w)
    print('Vocabulary size is {0}'.format(word_dim))

    output_type = 'discrete'
    if output_type == 'sequence':
        X = np.asarray([x[:-1] for x in dataset['data']])
        Y = np.asarray([y[1:] for y in dataset['data']])
        out_dim = word_dim
    elif output_type == 'real':
        X = np.asarray(dataset['data'])
        Y = np.asarray(dataset['ncomments'])
        out_dim = 1
    elif output_type == 'discrete':
        X = np.asarray(dataset['data'])
        Y = np.asarray(dataset['ncomments'])
        Y = (Y > 10).astype(int)
        out_dim = 2

    X_train, Y_train = X[:-100], Y[:-100]
    X_test, Y_test = X[-100:], Y[-100:]

    # It seems decoupling input to hidden is more important for extended RNN
    model = UGRNN(word_dim, 100, out_dim, output_type,
                    lmbda=0.0001, dropout=0.5,
                    decoupling=False, truncate_gradient=-1, borrow=False)

    cost_hist, ve_hist, te_hist = model.fit(X_train, Y_train,
        patience=3, valid_freq=500, valid_percent=0.3, valid_batch_size=256,
        batch_size=32, max_epoch=200, learning_rate=1e-3, max_len=None
    )

    plt.plot(te_hist[:, 0], te_hist[:, 1], 'r.',
             ve_hist[:, 0], ve_hist[:, 1], 'b')
    plt.xlabel('Number of updates')
    plt.ylabel('Prediction error')
    plt.show()

    # with gzip.open(model.__class__.__name__+'.model', mode='wb') as f:
    #     pickle.dump(model.params.get_value(), f, protocol=-1)

    # print(X_test[0])
    # for i, _ in enumerate(X_test[0]):
    #     if i > 0:
    #         x = np.array(X_test[0][:i], dtype=np.int64)[:, None]
    #         mask = 0 * x + 1.
    #         pred_char = model.predict(x, mask)
    #         print(i2w[X_test[0][i]], i2w[pred_char[-1][0]])

