from functools import singledispatch
from typing import Optional

import gpflow
import gpflux
import tensorflow as tf
from gpflow.base import RegressionData
from gpflow.monitor import Monitor

__all__ = ['run_optimizer']


class CallbackFromMonitor(tf.keras.callbacks.Callback):
    """Adapts a GPflow Monitor into a Keras Callback"""

    def __init__(self, monitor: Monitor):
        super().__init__()
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        self.monitor(epoch)

# !!IMPORTANT!! DUE TO NEWER TF VERSION, THIS HAD TO BE UPDATED AND REPLACED!
def run_iterative_opt(opt, loss_fn, vars, steps, monitor=None, first_step=0):
    for i in tf.range(steps):
        with tf.GradientTape() as tape:
            # Call loss function if needed
            loss_value = loss_fn() if callable(loss_fn) else loss_fn
        grads = tape.gradient(loss_value, vars)
        opt.apply_gradients(zip(grads, vars))

        if monitor is not None:
            monitor(first_step + i)



# noinspection PyUnusedLocal
@singledispatch
def run_optimizer(
        m: object, opt: tf.keras.optimizers.Optimizer,
        total_steps: int, initial_step: int = 0,
        monitor: Optional[Monitor] = None,
        train_data: Optional[RegressionData] = None
):
    raise NotImplementedError(f'No default behaviour for {type(m)!r}')


# noinspection PyUnusedLocal
@run_optimizer.register
def run_optimizer_for_gpflow(
        m: gpflow.models.GPModel, opt: tf.keras.optimizers.Optimizer,
        total_steps: int, initial_step: int = 0,
        monitor: Optional[Monitor] = None,
        train_data: Optional[RegressionData] = None
):
    # !!IMPORTANT!! Wrap training_loss so it’s bound to m
    loss_fn = lambda: m.training_loss()

    return tf.function(run_iterative_opt)(
        opt, loss_fn, m.trainable_variables, total_steps, monitor, initial_step
    )


@run_optimizer.register
def run_optimizer_for_keras(
        m: tf.keras.Model, opt: tf.keras.optimizers.Optimizer,
        total_steps: int, initial_step: int = 0,
        monitor: Optional[Monitor] = None,
        train_data: Optional[RegressionData] = None
):
    assert train_data is not None
    keras_callbacks = [CallbackFromMonitor(monitor)] if monitor is not None else None
    m.compile(opt)
    return m.fit(
        {"inputs": train_data[0], "targets": train_data[1]},
        batch_size=len(train_data[0]),
        epochs=initial_step + total_steps, initial_epoch=initial_step,
        verbose=0, callbacks=keras_callbacks
    )


@run_optimizer.register
def run_optimizer_for_gpflux(
        m: gpflux.models.DeepGP, opt: tf.keras.optimizers.Optimizer,
        total_steps: int, initial_step: int = 0,
        monitor: Optional[Monitor] = None,
        train_data: Optional[RegressionData] = None
):
    return run_optimizer_for_keras(
        m.as_training_model(), opt, total_steps, initial_step, monitor, train_data
    )
