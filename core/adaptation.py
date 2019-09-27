from enum import Enum, auto
from typing import Callable, Tuple

from tensorflow import Tensor, divide, identity
from tensorflow.python import shape, int32, cast
from tensorflow.python.keras import Model
from tensorflow.python.keras.activations import softmax
from tensorflow.python.keras.layers import Activation, concatenate

MetricType = Callable[[Tensor, Tensor], Tensor]


class Method(Enum):
    DISTILLATION = auto()
    PKT = auto()


def softmax_with_temperature(temperature: float) -> Callable[[Tensor], Tensor]:
    """
    Returns a softmax with temperature activation function.
    :param temperature: the temperature for the softmax operation.
    :return: Keras activation function.
    """

    def activation(x: Tensor) -> Tensor:
        """
        Function wrapped in order to create a Keras softmax activation function, which accepts a temperature parameter.
        :param x: the input tensor.
        :return: Tensor, output of softmax transformation (all values are non-negative and sum to 1).
        """
        if temperature == 1:
            result = softmax(x)
        else:
            result = softmax(divide(x, temperature))
        return result

    return activation


def kd_student_adaptation(model: Model, temperature: float) -> Model:
    """
    Adapts a student model for distillation.

    :param model: the model to be adapted for distillation.
    :param temperature: the temperature for the distillation process.
    :return: the adapted Model.
    """
    # Remove softmax.
    model.layers.pop()
    logits = model.layers[-1].output

    # Hard probabilities.
    probabilities = Activation('softmax')(logits)

    # Soft probabilities.
    if temperature == 1:
        probabilities_t = identity(probabilities)
    else:
        probabilities_t = Activation(softmax_with_temperature(temperature))(logits)

    outputs = concatenate([probabilities, probabilities_t])

    return Model(model.input, outputs)


def kd_student_rewind(model: Model) -> Model:
    """
    Rewinds an adapted student model for distillation, to its normal state.

    :param model: the model to be rewind.
    :return: the normal student Model.
    """
    # Remove concatenated layer.
    model.layers.pop()
    logits = model.layers[-1].output

    # Normal softmax probabilities.
    outputs = Activation('softmax')(logits)

    return Model(model.input, outputs)


def split_targets(y_true: Tensor, y_pred: Tensor, method: Method) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Split concatenated hard targets / logits and hard predictions / soft predictions.

    :param y_true: tensor with the true labels.
    :param y_pred: tensor with the predicted labels.
    :param method: the method used to transfer the knowledge.
    :return: the concatenated logits, soft predictions, hard targets and hard predictions
    (teacher_logits, student_output, y_true, y_pred).
    """
    # Here we get the split point, which is half of the predicting dimension.
    # The reason is because the network's output contains the predicted values
    # concatenated with the predicted logits, which will always have the same dimension.
    split_point = cast(divide(shape(y_true)[1], 2), int32)
    # Get hard labels and logits.
    y_true, teacher_logits = y_true[:, :split_point], y_true[:, split_point:]

    if method == Method.DISTILLATION:
        y_pred, student_output = y_pred[:, :split_point], y_pred[:, split_point:]
    else:
        student_output = identity(y_pred)

    return teacher_logits, student_output, y_true, y_pred


def kt_metric(metric_function: MetricType, method: Method) -> MetricType:
    """
    Creates a Keras metric function, which splits the predictions, to be used for evaluation.
    :param metric_function: the Keras metric function to adjust.
    :param method: the method used to transfer the knowledge.
    :return: the metric_function.
    """

    def metric(y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Function wrapped in order to create a Keras metric function, which splits the predictions.
        :param y_true: tensor with the true labels.
        :param y_pred: tensor with the predicted labels.
        :return: the metric_function.
        """
        _, _, y_true, y_pred = split_targets(y_true, y_pred, method)
        return metric_function(y_true, y_pred)

    return metric
