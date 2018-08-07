import numpy as np
from keras.layers import Layer
import keras.backend as K

"""
Hash embedding layer. Note that the zero word index is always used for masking.
    # Properties
        max_word_idx: maximum word index (e.g. the maximum dictionary value).
        num_buckets: number of buckets
        embedding_size: size of embedding
        num_hash_functions: number of hash functions
        W_trainable = True, if the embedding should be trainable
        p_trainable = True, if the importance parameters should be trainable
        append_weight, True if the importance parameters should be appended
        aggregation_mode: either 'sum' or 'concatenate' depending on whether
                        the component vectors should be summed or concatenated


"""
class HashEmbedding(Layer):
    def __init__(self, max_word_idx, num_buckets, embedding_size, num_hash_functions=2,
                 W_trainable=True, p_trainable = True, append_weight= True, aggregation_mode = 'sum', seed=3, **kwargs):
        super(HashEmbedding, self).__init__(**kwargs)
        np.random.seed(seed)
        self.word_count = max_word_idx
        W = np.random.normal(0, 0.1, (num_buckets, embedding_size))
        self.num_buckets = W.shape[0]
        self.mask_zero = True
        self.append_weight = append_weight
        self.p = None
        self.trainable_weights = []
        self.p_trainable = p_trainable
        self.num_hashes = num_hash_functions
        self.p_init_std = 0.0005

        self.num_hash_functions = num_hash_functions
        self.hashing_vals = []
        self.hashing_offset_vals = []


        # Initialize hash table. Note that this could easily be implemented by a modulo operation
        tab = (np.random.randint(0, 2 ** 30, size=(self.word_count, self.num_hash_functions)) % self.num_buckets) + 1
        self.hash_tables = K.variable(tab, dtype='int32')

        # Initialize word importance parameters
        p_init = np.random.normal(0, self.p_init_std, (self.word_count, self.num_hashes))
        self.p = K.variable(p_init,name='p_hash')
        if self.p_trainable:
            self.trainable_weights.append(self.p)


        #Initialize the embedding matrix
        # add zero vector for nulls (for masking)
        W = np.row_stack((np.zeros((1, W.shape[1])), W)).astype('float32')
        self.embedding_size = W.shape[1]
        W_shared = K.variable(W, name='W_hash')
        self.W = W_shared
        if W_trainable:
            self.trainable_weights.append(self.W)

        if aggregation_mode == 'sum':
            self.aggregation_function = sum
        else:
            if aggregation_mode == 'concatenate':
                self.aggregation_function = lambda x: K.concatenate(x,axis = -1)
            else:
                raise('unknown aggregation function')
        self.aggregation_mode = aggregation_mode

    def compute_mask(self, x, mask=None):
        if not self.mask_zero:
            return None
        else:
            return K.not_equal(x, 0)

    def call(self, input, mask=None):
        W = self.W
        pvals = []
        retvals = []
        input_w = input%self.word_count
        input_p = (3+input)%self.word_count
        idx_bucket_all = K.gather(self.hash_tables, input_w)
        for hash_fun_num in range(self.num_hash_functions):
            W0 = K.gather(W, idx_bucket_all[:,:,hash_fun_num]*(1-K.cast(K.equal(0, input_w), 'int32')))
            p_0 = K.gather(self.p[:,hash_fun_num], input_p)
            p = K.expand_dims(p_0,dim=-1)
            pvals.append(p)
            retvals.append(W0*p)
        retval = self.aggregation_function(retvals)
        if self.append_weight:
            retval = K.concatenate([retval]+pvals,axis=-1)
        return retval


    def get_output_shape_for(self, input_shape):
        weight_addition = 0
        if self.append_weight:
            weight_addition = self.num_hash_functions
        if self.aggregation_mode == 'sum':
            return (input_shape[0], input_shape[1], self.embedding_size+weight_addition)
        else:
            return (input_shape[0], input_shape[1], self.embedding_size * self.num_hash_functions + weight_addition)


class ReduceSum(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True

        super(ReduceSum, self).__init__(**kwargs)

    def call(self, x, mask=None):
        x, m = x
        x = x * K.cast(K.expand_dims(K.not_equal(m,0), -1), 'float32')
        x = K.cast(x, 'float32')
        return K.sum(x, axis=1,keepdims=False)

    def compute_mask(self, input, mask):
        return None

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], input_shape[0][2])