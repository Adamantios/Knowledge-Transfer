from tensorflow import Tensor
from tensorflow.python import constant
from tensorflow.python.keras.backend import dot, sum
from tensorflow.python.keras.losses import categorical_crossentropy, kullback_leibler_divergence
from tensorflow.python.ops.math_ops import divide
from tensorflow.python.ops.nn_ops import softmax


def supervised_loss(y_true: Tensor, y_pred: Tensor, lambda_const: float) -> Tensor:
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


def distillation_loss(teacher_logits: Tensor, student_logits: Tensor, temperature: float,
                      lambda_const: float) -> Tensor:
    """
    Calculates the Distillation Loss between two networks' logits.

    :param teacher_logits: the teacher network's logits.
    :param student_logits: the student network's logits.
    :param temperature: the temperature for the softmax.
    :param lambda_const: the importance weight of the supervised loss.
    :return: the distillation loss.
    """
    y_teacher = softmax(divide(teacher_logits, temperature))
    y_student = softmax(divide(student_logits, temperature))
    loss = categorical_crossentropy(y_teacher, y_student)
    return loss + supervised_loss(y_teacher, y_student, lambda_const)


def pkt_loss(y_teacher: Tensor, y_student: Tensor) -> Tensor:
    """
    Calculates the Probabilistic Knowledge Transfer Loss between two networks.

    :param y_teacher: the teacher's values.
    :param y_student: the student's values.
    :return: the transfer loss.
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
