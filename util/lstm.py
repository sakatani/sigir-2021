import tensorflow as tf

class Linear(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, activation='relu'):
        super().__init__()
        self.dense = tf.keras.layers.Dense(hidden_dim)
        self.norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation(activation)

    def call(self, input_):
        x = self.dense(input_)
        x = self.norm(x)
        x = self.activation(x)
        return x

def lstm_model(features, embedding_map, max_embed_dim, hidden_dim, target_count, n_top=20):
    inputs = tf.keras.layers.Input(shape=(len(features),))
    embs = []
    dim_i, dim_o = embedding_map['product_hash_lag1']
    embedding_product = tf.keras.layers.Embedding(dim_i, dim_o)
    dim_i, dim_o = embedding_map['category_lag1']
    embedding_category = tf.keras.layers.Embedding(dim_i, dim_o)
    product_embs = []
    cat_embs = []
    # url_embs = []

    for k, f in enumerate(features):
        dim_i, dim_o = embedding_map[f]
        if f.startswith('product_hash'):
            product_embs.append(embedding_product(inputs[:, k]))
        elif f.startswith('category_'):
            cat_embs.append(embedding_category(inputs[:, k]))
        # elif f.startswith('url_'):
        #     url_embs.append(e_url(inp[:,k]))
        else:
            embedding_tmp = tf.keras.layers.Embedding(dim_i, dim_o)
            embs.append(embedding_tmp(inputs[:, k]))

    xp = tf.stack(product_embs, axis=1)
    xp = tf.keras.layers.LSTM(max_embed_dim, activation='tanh')(xp)

    xc = tf.stack(cat_embs, axis=1)
    _, dim_o = embedding_map['category_lag1']
    xc = tf.keras.layers.LSTM(dim_o, activation='tanh')(xc)

    # xu = tf.stack(url_embs, axis=1)
    # xu = tf.keras.layers.LSTM(EC, activation='tanh')(xu)

    x = tf.keras.layers.Concatenate()(embs)
    x = tf.keras.layers.Concatenate()([x, xp, xc]), # , xu])

    x = Linear(hidden_dim)(x)
    x = Linear(hidden_dim)(x)
    prob = tf.keras.layers.Dense(target_count, activation='softmax', name='main_output')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=[prob])
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    mtr = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=n_top)
    model.compile(loss=loss, optimizer=optimizer, metrics=[mtr])
    return model