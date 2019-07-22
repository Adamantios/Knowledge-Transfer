import numpy as np
from tensorflow import Tensor
from tensorflow.python import constant
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.ops.nn_ops import softmax


def supervised_loss(y_true: np.ndarray, y_pred: np.ndarray, lambda_const: float) -> Tensor:
    """
    Calculates the hard targets log-loss.
    :param y_true: the true labels.
    :param y_pred: the predicted labels.
    :param lambda_const: the importance weight of the supervised loss.
    :return: the supervised log-loss.
    """
    if lambda_const:
        return lambda_const * categorical_crossentropy(y_true, y_pred)
    else:
        return constant(0)


def distillation_loss(teacher_logits: np.ndarray, student_logits: np.ndarray, temperature: float,
                      lambda_const: float) -> Tensor:
    """
    Calculates the Distillation Loss between two networks logits.

    :param teacher_logits: the teacher network's logits.
    :param student_logits: the student network's logits.
    :param temperature: the temperature for the softmax.
    :param lambda_const: the importance weight of the supervised loss.
    :return: the distillation loss.
    """
    y_teacher = softmax(teacher_logits / temperature)
    y_student = softmax(student_logits / temperature)
    loss = categorical_crossentropy(y_teacher, y_student)
    return loss + supervised_loss(y_teacher, y_student, lambda_const)
