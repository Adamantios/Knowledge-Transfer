from tensorflow import Tensor, transpose, shape
from tensorflow.python import constant, less, while_loop, zeros
from tensorflow.python.keras.losses import categorical_crossentropy, kullback_leibler_divergence
from tensorflow.python.ops.math_ops import multiply, add, reduce_sum, divide, matmul
from tensorflow.python.ops.nn_impl import l2_normalize

from core.adaptation import MetricType, split_targets, softmax_with_temperature, Method

LossType = MetricType


# TODO add cycle loss.
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
    if bool(lambda_const):
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


def _calculate_supervised_similarities(y_true) -> Tensor:
    """
    Calculates the target supervised similarities.
    Performs a tensorflow nested loop, in order to compare the values of y_true for range(batch_size).

    :param y_true: the y_true value.
    :return: Tensor containing the target supervised similarities.
    """
    # Get the batch size.
    batch_size = shape(y_true)[0]
    # Initialize outer loop index.
    i = constant(0)
    # Initialize symmetric supervised similarity matrix targets.
    target_similarity = zeros((batch_size, batch_size))

    def outer_loop_condition(_i, _batch_size, _y_true, _target_similarity):
        """Define outer loop condition."""
        return less(_i, _batch_size)

    def outer_loop_body(_i, _batch_size, _y_true, _target_similarity):
        """Define outer loop body."""
        # Initialize inner loop index.
        j = constant(0)

        def inner_loop_condition(_i, _j, _y_true, _target_similarity):
            """Define inner loop condition."""
            return less(_j, _batch_size)

        def inner_loop_body(_i, _j, _y_true, _target_similarity):
            """Define inner loop body."""
            if _y_true[_i] == _y_true[_j]:
                _target_similarity[_i, _j] = 1
            return _i, _j + 1, _y_true, _target_similarity

        # Begin inner while loop.
        _, j, _, _target_similarity = while_loop(inner_loop_condition, inner_loop_body,
                                                 [_i, j, _y_true, _target_similarity])
        return _i + 1, _batch_size, _y_true, _target_similarity

    # Begin outer while loop.
    i, _, _, target_similarity = while_loop(outer_loop_condition, outer_loop_body,
                                            [i, batch_size, y_true, target_similarity])
    return target_similarity


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
        l2_normalize(tensor, axis=1)
        return matmul(tensor, transpose(tensor))

    def to_probabilities(tensor: Tensor):
        """ Transforms a symmetric 2D array's values into probabilities. """
        return divide(tensor, reduce_sum(tensor, axis=1, keepdims=True))

    teacher_similarity, student_similarity = cosine_similarity(y_teacher), cosine_similarity(y_student)
    teacher_similarity, student_similarity = to_probabilities(teacher_similarity), to_probabilities(student_similarity)
    loss = kullback_leibler_divergence(teacher_similarity, student_similarity)

    # If supervised transfer is being performed.
    if bool(lambda_const):
        target_similarity = _calculate_supervised_similarities(y_true)
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
