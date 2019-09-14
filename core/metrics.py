from itertools import combinations_with_replacement
from typing import Callable, Tuple

from tensorflow import Tensor, zeros
from tensorflow.python.keras.backend import dot, sum, identity
from tensorflow.python.keras.losses import categorical_crossentropy, kullback_leibler_divergence
from tensorflow.python.ops.math_ops import divide, multiply, add
from tensorflow.python.ops.nn_ops import softmax

LossType = MetricType = Callable[[Tensor, Tensor], Tensor]


def _split_targets(y_true: Tensor, y_pred: Tensor, hard_targets_exist: bool) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Split concatenated hard targets and logits.

    :param y_true: tensor with the true labels.
    :param y_pred: tensor with the predicted labels.
    :param hard_targets_exist: whether the hard targets exist or not.
    :return: the hard targets and the logits (teacher_logits, student_logits, y_true, y_pred).
    """
    if hard_targets_exist:
        # Here we get the split point, which is half of the predicting dimension.
        # The reason is because the network's output contains the predicted values
        # concatenated with the predicted logits, which will always have the same dimension.
        split_point = y_true.get_shape().as_list()[1] / 2
        # Get hard labels and logits.
        y_true, teacher_logits = y_true[:, :split_point], y_true[:, split_point:]
        y_pred, student_logits = y_pred[:, :split_point], y_pred[:, split_point:]
    else:
        teacher_logits, student_logits = identity(y_true), identity(y_pred)
        y_true, y_pred = None, None

    return teacher_logits, student_logits, y_true, y_pred


def _distillation_loss_calculator(teacher_logits: Tensor, student_logits: Tensor, temperature: float,
                                  y_true: Tensor, y_pred: Tensor, lambda_const: float) -> Tensor:
    """
    Calculates the Distillation Loss between two networks.

    :param teacher_logits: the teacher network's logits.
    :param student_logits: the student network's logits.
    :param temperature: the temperature for the softmax.
    :param y_true: the true labels, if performing supervised distillation.
    :param y_pred: the predicted labels, if performing supervised distillation.
    :param lambda_const: the importance weight of the supervised loss.
    Set it to 0 if you do not want to apply supervised loss.
    :return: the distillation loss.
    """
    # Apply softmax with temperature to the teacher's logits.
    y_teacher = softmax(divide(teacher_logits, temperature))
    # Calculate log-loss.
    loss = categorical_crossentropy(y_teacher, student_logits)

    # If supervised distillation is being performed, add supervised loss, multiplied by its importance weight.
    if lambda_const:
        loss = add(loss, multiply(lambda_const, categorical_crossentropy(y_true, y_pred)))

    return loss


def distillation_loss(temperature: float, lambda_const: float) -> LossType:
    """
    Calculates the Distillation Loss between two networks.

    :param temperature: the temperature for the softmax.
    :param lambda_const: the importance weight of the supervised loss.
    Set it to 0 if you do not want to apply supervised loss.
    :return: the distillation loss.
    """

    def distillation(y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Function wrapped, in order to create a Keras Distillation Loss function.
        :param y_true: tensor with the true labels.
        :param y_pred: tensor with the predicted labels.
        :return: the distillation loss.
        """
        teacher_logits, student_logits, y_true, y_pred = _split_targets(y_true, y_pred, bool(lambda_const))
        return _distillation_loss_calculator(teacher_logits, student_logits, temperature, y_true, y_pred, lambda_const)

    return distillation


def _pkt_loss_calculator(y_teacher: Tensor, y_student: Tensor, y_true: Tensor, y_pred: Tensor,
                         lambda_const: float) -> Tensor:
    """
    Calculates the Probabilistic Knowledge Transfer Loss between two networks.

    :param y_teacher: the teacher's values.
    :param y_student: the student's values.
    :param y_true: the true labels, if performing supervised distillation.
    :param y_pred: the predicted labels, if performing supervised distillation.
    :param lambda_const: the importance weight of the supervised loss.
    Set it to 0 if you do not want to apply supervised loss.
    :return: the probabilistic knowledge transfer loss.
    """

    def cosine_similarity(tensor: Tensor) -> Tensor:
        """ Calculates the cosine similarity of a 2D array. """
        return dot((tensor, tensor.T), l2_normalize=True)

    def to_probabilities(tensor: Tensor):
        """ Transforms a symmetric 2D array's values into probabilities. """
        return tensor / sum(tensor, axis=1, keepdims=True)

    teacher_similarity, student_similarity = cosine_similarity(y_teacher), cosine_similarity(y_student)
    teacher_similarity, student_similarity = to_probabilities(teacher_similarity), to_probabilities(student_similarity)
    loss = kullback_leibler_divergence(teacher_similarity, student_similarity)

    # If supervised distillation is being performed.
    if lambda_const:
        # Initialize symmetric supervised similarity matrix targets.
        target_similarity = zeros(y_true.shape[0], y_true.shape[0])
        # Run through all the target index combinations, without duplicates.
        for i, j in combinations_with_replacement(y_true.shape[0], 2):
            # If samples have the same target, make symmetric similarity 1.
            if y_true[i] == y_true[j]:
                target_similarity[i, j] = target_similarity[j, i] = 1

        predicted_similarity = cosine_similarity(y_pred)
        target_similarity = to_probabilities(target_similarity)
        predicted_similarity = to_probabilities(predicted_similarity)

        # Add supervised loss, multiplied by its importance weight.
        add(loss, multiply(lambda_const, kullback_leibler_divergence(target_similarity, predicted_similarity)))

    return loss


def pkt_loss(lambda_const: float) -> LossType:
    """
    Calculates the Probabilistic Knowledge Transfer Loss between two networks.

    :param lambda_const: the importance weight of the supervised loss.
    Set it to 0 if you do not want to apply supervised loss.
    :return: the probabilistic knowledge transfer loss.
    """

    def pkt(y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Function wrapped, in order to create a Keras Probabilistic Knowledge Transfer Loss function.
        :param y_true: tensor with the true labels.
        :param y_pred: tensor with the predicted labels.
        :return: the probabilistic knowledge transfer loss.
        """
        teacher_logits, student_logits, y_true, y_pred = _split_targets(y_true, y_pred, bool(lambda_const))
        return _pkt_loss_calculator(teacher_logits, student_logits, y_true, y_pred, lambda_const)

    return pkt


available_losses = [
    {
        'name': 'Distillation Loss',
        'function': distillation_loss
    },
    {
        'name': 'PKT Loss',
        'function': pkt_loss
    }
]


def kt_metric(hard_targets_exist: bool, metric_function: MetricType) -> MetricType:
    """
    Creates a Keras metric function, which splits the predictions.
    :param hard_targets_exist: whether the hard targets exist or not.
    :param metric_function: the Keras metric function to adjust.
    :return: the metric_function.
    """

    def metric(y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Function wrapped, in order to create a Keras metric function, which splits the predictions.
        :param y_true: tensor with the true labels.
        :param y_pred: tensor with the predicted labels.
        :return: the metric_function.
        """
        _, _, y_true, y_pred = _split_targets(y_true, y_pred, hard_targets_exist)
        return metric_function(y_true, y_pred)

    return metric
