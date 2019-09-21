from itertools import combinations_with_replacement

from tensorflow import Tensor, zeros
from tensorflow.python.keras.backend import dot, sum
from tensorflow.python.keras.losses import categorical_crossentropy, kullback_leibler_divergence
from tensorflow.python.ops.math_ops import multiply, add

from core.adaptation import MetricType, split_targets, softmax_with_temperature, Method

LossType = MetricType


def _distillation_loss_calculator(teacher_logits: Tensor, y_student: Tensor, temperature: float,
                                  y_true: Tensor, y_pred: Tensor, lambda_const: float) -> Tensor:
    """
    Calculates the Distillation Loss between two networks.

    :param teacher_logits: the teacher network's logits.
    :param y_student: the student network's output.
    :param temperature: the temperature for the softmax.
    :param y_true: the true labels, if performing supervised distillation.
    :param y_pred: the predicted labels, if performing supervised distillation.
    :param lambda_const: the importance weight of the supervised loss.
    Set it to 0 if you do not want to apply supervised loss.
    :return: the distillation loss.
    """
    # Apply softmax with temperature to the teacher's logits.
    y_teacher = softmax_with_temperature(temperature)(teacher_logits)
    # Calculate log-loss.
    loss = categorical_crossentropy(y_teacher, y_student)

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
        teacher_logits, student_output, y_true, y_pred = split_targets(y_true, y_pred, Method.DISTILLATION)
        return _distillation_loss_calculator(teacher_logits, student_output, temperature, y_true, y_pred, lambda_const)

    return distillation


def _pkt_loss_calculator(y_teacher: Tensor, y_student: Tensor, y_true: Tensor, lambda_const: float) -> Tensor:
    """
    Calculates the Probabilistic Knowledge Transfer Loss between two networks.

    :param y_teacher: the teacher's values.
    :param y_student: the student's values.
    :param y_true: the true labels, if performing supervised distillation.
    :param lambda_const: the importance weight of the supervised loss.
    Set it to 0 if you do not want to apply supervised loss.
    :return: the probabilistic knowledge transfer loss.
    """

    def cosine_similarity(tensor: Tensor) -> Tensor:
        """ Calculates the cosine similarity of a 2D array, with l2 normalization. """
        return dot((tensor, tensor.T), l2_normalize=True)

    def to_probabilities(tensor: Tensor):
        """ Transforms a symmetric 2D array's values into probabilities. """
        return tensor / sum(tensor, axis=1, keepdims=True)

    teacher_similarity, student_similarity = cosine_similarity(y_teacher), cosine_similarity(y_student)
    teacher_similarity, student_similarity = to_probabilities(teacher_similarity), to_probabilities(student_similarity)
    loss = kullback_leibler_divergence(teacher_similarity, student_similarity)

    # If supervised transfer is being performed.
    if lambda_const:
        # Initialize symmetric supervised similarity matrix targets.
        target_similarity = zeros(y_true.shape[0], y_true.shape[0])
        # Run through all the target index combinations, without duplicates.
        for i, j in combinations_with_replacement(y_true.shape[0], 2):
            # If samples have the same target, make symmetric similarity 1.
            if y_true[i] == y_true[j]:
                target_similarity[i, j] = target_similarity[j, i] = 1

        target_similarity = to_probabilities(target_similarity)

        # Add supervised loss, multiplied by its importance weight.
        add(loss, multiply(lambda_const, kullback_leibler_divergence(target_similarity, student_similarity)))

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
        teacher_output, student_output, y_true, _ = split_targets(y_true, y_pred, Method.PKT)
        return _pkt_loss_calculator(teacher_output, student_output, y_true, lambda_const)

    return pkt
