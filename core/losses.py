from typing import Callable, Tuple

from tensorflow import Tensor
from tensorflow.python.keras.backend import dot, sum, identity
from tensorflow.python.keras.losses import categorical_crossentropy, kullback_leibler_divergence
from tensorflow.python.ops.math_ops import divide, multiply, add
from tensorflow.python.ops.nn_ops import softmax

LossType = Callable[[Tensor, Tensor], Tensor]


def _split_targets(y_true: Tensor, y_pred: Tensor, split_point: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Split concatenated hard targets and logits.

    :param y_true: tensor with the true labels.
    :param y_pred: tensor with the predicted labels.
    :param split_point: the point in which to split the targets.
    :return: the hard targets and the logits (teacher_logits, student_logits, y_true, y_pred).
    """
    # If no split point has been set, init logits only.
    if split_point is None:
        teacher_logits, student_logits = identity(y_true), identity(y_pred)
        y_true, y_pred = None, None
    # Otherwise, get hard labels and logits.
    else:
        y_true, teacher_logits = y_true[:, :split_point], y_true[:, split_point:]
        y_pred, student_logits = y_pred[:, :split_point], y_pred[:, split_point:]

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


def distillation_loss(temperature: float, split_point: int, lambda_const: float) -> LossType:
    """
    Calculates the Distillation Loss between two networks.

    :param temperature: the temperature for the softmax.
    :param split_point: the point where the hard targets will be split from the soft targets.
    Set to None if performing unsupervised distillation.
    :param lambda_const: the importance weight of the supervised loss.
    :return: the distillation loss.
    """

    def distillation(y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Function wrapped, in order to create a Keras Distillation Loss function.
        :param y_true: tensor with the true labels.
        :param y_pred: tensor with the predicted labels.
        :return: the distillation loss.
        """
        teacher_logits, student_logits, y_true, y_pred = _split_targets(y_true, y_pred, split_point)
        return _distillation_loss_calculator(teacher_logits, student_logits, temperature, y_true, y_pred, lambda_const)

    return distillation


def _pkt_loss_calculator(y_teacher: Tensor, y_student: Tensor) -> Tensor:
    """
    Calculates the Probabilistic Knowledge Transfer Loss between two networks.

    :param y_teacher: the teacher's values.
    :param y_student: the student's values.
    :return: the probabilistic knowledge transfer loss.
    """
    # Calculate the cosine similarities.
    # TODO check if .T is the same as .transpose(0, 1).
    y_teacher = dot((y_teacher, y_teacher.transpose(0, 1)), l2_normalize=True)
    y_student = dot((y_student, y_student.transpose(0, 1)), l2_normalize=True)

    # Transform them into probabilities.
    y_teacher = y_teacher / sum(y_teacher, dim=1, keepdim=True)
    y_student = y_student / sum(y_student, dim=1, keepdim=True)

    # Calculate the KL-divergence.
    loss = kullback_leibler_divergence(y_teacher, y_student)

    return loss


def pkt_loss(split_point: int, lambda_const: float) -> LossType:
    """
    Calculates the Probabilistic Knowledge Transfer Loss between two networks.

    :param split_point: the point where the hard targets will be split from the soft targets.
    Set to None if performing unsupervised knowledge transfer.
    :param lambda_const: the importance weight of the supervised loss.
    :return: the probabilistic knowledge transfer loss.
    """

    def pkt(y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        Function wrapped, in order to create a Keras Probabilistic Knowledge Transfer Loss function.
        :param y_true: tensor with the true labels.
        :param y_pred: tensor with the predicted labels.
        :return: the probabilistic knowledge transfer loss.
        """
        teacher_logits, student_logits, y_true, y_pred = _split_targets(y_true, y_pred, split_point)
        loss = _pkt_loss_calculator(teacher_logits, student_logits)

        if lambda_const:
            add(loss, multiply(lambda_const, _pkt_loss_calculator(y_true, y_pred)))

        return loss

    return pkt
