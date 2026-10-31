import tensorflow as tf
from tensorflow.keras import layers, models, Input
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

# Pre-norm version (this version is deprecated)
def transformer_encoder_prenorm(x, head_dim, num_heads, ff_dim, dropout=0.1):
    # ===== Attention block (Pre-Norm) =====
    h = layers.LayerNormalization(epsilon=1e-6)(x)
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_dim,
        dropout=dropout
    )(h, h)
    x = layers.Add()([x, attn])  # residual

    # ===== FFN block (Pre-Norm) =====
    h = layers.LayerNormalization(epsilon=1e-6)(x)
    ff = layers.Dense(ff_dim, activation="relu")(h)
    ff = layers.Dense(x.shape[-1])(ff)
    x = layers.Add()([x, ff])  # residual

    return x

# Resnet-style MLP block (deprecated)
def residual_mlp_block(x, hidden_dim):
    h = layers.LayerNormalization(epsilon=1e-6)(x)

    h = layers.Dense(hidden_dim, activation="relu")(h)
    h = layers.Dense(x.shape[-1])(h)

    return layers.Add()([x, h])

# Currently used transformer block**
def transformer_encoder(x, head_dim, num_heads, ff_dim, dropout=0.1):
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=head_dim,
        dropout=dropout
    )(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(x.shape[-1])(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    return x

# loss function
def gaussian_nll(y_true, y_pred):
    def split_mu_sigma(y_pred):
        mu = y_pred[:, 0:1]
        log_sigma = y_pred[:, 1:2]  # Using log σ is more stable
        sigma = tf.exp(log_sigma)
        return mu, sigma
    mu, sigma = split_mu_sigma(y_pred)

    loss = tf.math.log(sigma) + tf.square(y_true - mu) / (2.0 * tf.square(sigma))
    return tf.reduce_mean(loss)

def build_MD_trans2(NumSpecies):

    inp_g = Input(shape=(NumSpecies,), name="growth")
    inp_A = Input(shape=(NumSpecies, NumSpecies), name="interaction")
    inp_d = Input(shape=(NumSpecies,), name="dispersal")

    g_exp = layers.Reshape((NumSpecies, 1))(inp_g)
    d_exp = layers.Reshape((NumSpecies, 1))(inp_d)

    tokens = layers.Concatenate(axis=-1)([g_exp, d_exp, inp_A])
    tokens = layers.Dense(64)(tokens)

    x = transformer_encoder(tokens, head_dim=16, num_heads=4, ff_dim=128)
    x = transformer_encoder(x, head_dim=16, num_heads=4, ff_dim=128)

    h = layers.GlobalAveragePooling1D()(x)

    h = layers.Dense(64, activation="relu")(h)
    h = layers.Dense(128, activation="relu")(h)
    h = layers.Dense(64, activation="relu")(h)

    out_y = layers.Dense(1, activation="linear")(h)

    model = models.Model(
        inputs=[inp_g, inp_A, inp_d],
        outputs=out_y
    )

    model.compile(optimizer="adam", loss="mse")

    return model

def build_MD_trans3(NumSpecies):
    # ---------- inputs ----------
    inp_g = Input(shape=(NumSpecies,), name="growth")
    inp_A = Input(shape=(NumSpecies, NumSpecies), name="interaction")
    inp_d = Input(shape=(NumSpecies,), name="dispersal")

    # ---------- interaction → Transformer ----------
    g_exp = layers.Lambda(lambda x: tf.expand_dims(x, -1))(inp_g)
    d_exp = layers.Lambda(lambda x: tf.expand_dims(x, -1))(inp_d)

    # token_i = [g_i, d_i, A_i1...A_iN]
    tokens = layers.Concatenate(axis=-1)([g_exp, d_exp, inp_A])
    tokens = layers.Dense(64)(tokens) # embedding

    x = transformer_encoder_prenorm(tokens, head_dim=16, num_heads=4, ff_dim=128)
    x = transformer_encoder_prenorm(x, head_dim=16, num_heads=4, ff_dim=128)

    h = layers.GlobalAveragePooling1D()(x)

    # # First unify dimensions (very important), then stack several residual blocks
    # h = layers.Dense(64, activation="relu")(h)
    # h = residual_mlp_block(h, hidden_dim=128)

    # ---------- output ----------
    out_y = layers.Dense(1, activation="linear", name="y")(h)

    model = models.Model(
        inputs=[inp_g, inp_A, inp_d],
        outputs=out_y
    )

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model

def weighted_bce(pos_weight=10.0):

    def loss(y_true, y_pred):

        bce = K.binary_crossentropy(y_true, y_pred)

        weight = 1 + y_true * (pos_weight - 1)

        return K.mean(weight * bce)

    return loss

def build_SS_trans2(NumSpecies):

    inp_g = Input(shape=(NumSpecies,), name="growth")
    inp_A = Input(shape=(NumSpecies, NumSpecies), name="interaction")
    inp_d = Input(shape=(NumSpecies,), name="dispersal")

    g_exp = layers.Reshape((NumSpecies, 1))(inp_g)
    d_exp = layers.Reshape((NumSpecies, 1))(inp_d)

    tokens = layers.Concatenate(axis=-1)([g_exp, d_exp, inp_A])
    tokens = layers.Dense(64)(tokens)

    x = transformer_encoder(tokens, head_dim=16, num_heads=4, ff_dim=128)
    x = transformer_encoder(x, head_dim=16, num_heads=4, ff_dim=128)
    x = layers.Dense(64, activation="gelu")(x)
    x = layers.Dropout(0.2)(x)

    out_y = layers.Dense(1, activation="sigmoid")(x)
    out_y = layers.Reshape((NumSpecies,))(out_y)

    model = models.Model(
        inputs=[inp_g, inp_A, inp_d],
        outputs=out_y
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy"
    )

    # model.compile(
    #     optimizer="adam",
    #     loss=weighted_bce(pos_weight=3.0),
    # )
    
    return model

def plot_training_history(history):
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    #plt.ylim(0, 0.01)
    plt.legend()
    plt.tight_layout()
    plt.show()