from tensorflow.keras import Model as KModel
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.metrics import RootMeanSquaredError, KLDivergence
from typing import Tuple
from chess_rules.chess_pi import ChessPI
from chess_rules.chess_state import ChessState
from model import Model
from pi import PI
from state import State
from config_loader import config
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dense, Softmax, ReLU, Add, Flatten
from tensorflow.keras.regularizers import l2
from logger import get_logger
import numpy as np

lambda_val = config["l2_regularization"]

logger = get_logger(__name__)

def conv_block(x, f=256, k=3, activate=True, normalize=True, kernel_init="he_uniform"):
    x = Conv2D(
        f, 
        k, 
        padding="same", 
        kernel_initializer=kernel_init, 
        kernel_regularizer=l2(lambda_val) 
    )(x)

    if normalize:
        x = BatchNormalization()(x)
    if activate:
        x = ReLU()(x)
    return x

def res_block(x):
    skip_x = x
    x = conv_block(x)
    x = conv_block(x, activate=False)
    x = Add()((x, skip_x))
    x = ReLU()(x)
    return x

def create_model(input_shape):
    input_layer = Input(shape=input_shape)
    x = input_layer
    x = conv_block(x)
    for _ in range(19):
        x = res_block(x)

    v_x = conv_block(x, 1, 1)
    v_x = Flatten()(v_x)
    v_x = Dense(
        256, 
        activation="linear", 
        kernel_regularizer=l2(lambda_val), 
        # activity_regularizer=l1(lambda_val)
    )(v_x)
    v_out = Dense(
        1,
        activation="tanh",
        kernel_regularizer=l2(lambda_val),
        name="v_out"
    )(v_x)

    pi_x = conv_block(x)
    pi_x = conv_block(pi_x, 73, activate=False, normalize=False, kernel_init="glorot_uniform")
    pi_out = Softmax(name="pi_out")(pi_x)

    model = KModel(inputs=input_layer, outputs=[pi_out, v_out])
    model.summary(print_fn=logger.debug)
    return model

def get_pi_loss(): 
    def loss(pi_true: tf.Tensor, p_pred: tf.Tensor):
        p_loss = tf.losses.poisson(pi_true, p_pred)
        return p_loss - tf.reduce_mean(p_pred)

    return loss

def get_v_loss():
    return "mse"

class ChessModel(Model):
    def __init__(self):
        self._model = create_model(ChessState.get_shape())
        self._model.compile(
            optimizer=Adam(
                learning_rate = config["model_learning_rate"]
            ),
            
            loss = {
                "pi_out": get_pi_loss(),
                "v_out": get_v_loss()
            },

            metrics = {
                "pi_out": KLDivergence(),
                "v_out": RootMeanSquaredError()
            }
        )

    def predict_pi_v(self, state: State) -> Tuple[PI, float]:
        state_nn = np.array([state.get_nn_rep()])
        pis, vs = self._model.predict(state_nn)
        return ChessPI.from_pi_dist(pis[0]), float(vs[0])

    @property
    def name(self) -> str:
        return "chess_model"

    def save(self, file_path: str, name: str):
        full_file_path = self.get_full_filepath(file_path, name)
        self._model.save(full_file_path)

    def load(self, file_path: str, name: str):
        full_file_path = self.get_full_filepath(file_path, name)
        self._model = load_model(full_file_path)

    def clone(self) -> Model:
        new_model = ChessModel()
        new_model._model = clone_model(self._model)
        return new_model

    def train(self, train_data: Model.DataType, test_data: Model.DataType):
        verbosity = config["model_verbosity"]
        epochs = config["model_epochs"]
        batch_size = config["model_batch_size"]
        X_train, Y_train = self.data_to_np(train_data)
        X_test, Y_test = self.data_to_np(test_data)
        self._model.fit(
            x=X_train,
            y=Y_train,
            validation_data=(X_test, Y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbosity
        )