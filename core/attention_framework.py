from typing import Tuple

from numpy.core.multiarray import ndarray
from tensorflow import zeros_like
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Concatenate, Activation, Dense, Multiply


def _pyramid_ensemble_adaptation(teacher: Model) -> Model:
    """
    Adapt pyramid ensemble by changing its output to contain each of its submodels outputs.

    :param teacher: the pyramid ensemble.
    :return: the attention pyramid ensemble.
    """
    # Get each submodel's outputs.
    output1 = teacher.get_layer('submodel_strong_output').output
    weak_1_output = teacher.get_layer('submodel_weak_1_output').output
    weak_2_output = teacher.get_layer('submodel_weak_2_output').output
    # Append zeros to the model outputs which do not predict all the classes.
    output2 = Concatenate(name='submodel_weak_1_output_fixed')([weak_1_output, zeros_like(weak_1_output)])
    output3 = Concatenate(name='submodel_weak_2_output_fixed')([zeros_like(weak_2_output), weak_2_output])
    # Add activations to the outputs.
    output1 = Activation('softmax', name='softmax1')(output1)
    output2 = Activation('softmax', name='softmax2')(output2)
    output3 = Activation('softmax', name='softmax3')(output3)
    # Concatenate submodels outputs.
    outputs = Concatenate(name='concatenated_submodels_outputs')([output1, output2, output3])

    # Create attention teacher.
    attention_teacher = Model(teacher.input, outputs, name='attention_' + teacher.name)
    return attention_teacher


def _complicated_ensemble_adaptation(teacher: Model) -> Model:
    """
    Adapt complicated ensemble by changing its output to contain each of its submodels outputs.

    :param teacher: the complicated ensemble.
    :return: the attention complicated ensemble.
    """
    # TODO
    pass


def _ensemble_adaptation(teacher: Model) -> Model:
    """
    Adapt an averaged predictions ensemble by changing its output to contain each of its submodels outputs.

    :param teacher: the ensemble.
    :return: the attention ensemble.
    """
    # TODO
    pass


def _teacher_adaptation(teacher: Model) -> Model:
    """
    Adapt teacher by changing its output to contain each of the submodels outputs.

    :param teacher: the teacher to adapt.
    :return: the attention teacher.
    """
    if 'pyramid_ensemble' in teacher.name:
        attention_teacher = _pyramid_ensemble_adaptation(teacher)
    elif 'complicated_ensemble' in teacher.name:
        attention_teacher = _complicated_ensemble_adaptation(teacher)
    elif teacher.name == 'ensemble':
        attention_teacher = _ensemble_adaptation(teacher)
    else:
        raise ValueError('Unknown teacher model has been given.')

    return attention_teacher


def _student_adaptation(student: Model, input_shape: tuple) -> Model:
    """
    Adjust student by adding a sidewalk for attention mechanism,
    in order to pay attention to each model of the teacher ensemble and create attention_student.

    :param student: the student to be based on.
    :param input_shape: the attention student's input shape.
    :return: the attention student.
    """
    # Initialize the student's input and the attention sidewalk's input, having the same shape as the teacher's output.
    student_input = Input(student.input_shape[1:], name='student_input')
    attention_inputs = Input(input_shape, name='attention_input')
    # Create attention vector.
    attention_vector = Dense(student.output_shape[1], activation='softmax', name='attention_vector')(attention_inputs)
    # Multiply attention values with student's outputs.
    outputs = Multiply(name='attention_weighted_predictions')([student(student_input), attention_vector])
    # Add a softmax.
    outputs = Activation('softmax', name='attention_weighted_predictions_softmax')(outputs)

    # Create and return attention student.
    attention_student = Model([student_input, attention_inputs], outputs, name='attention_' + student.name)
    return attention_student


def attention_framework_adaptation(x_train: ndarray, x_val: ndarray, teacher: Model, student: Model,
                                   evaluation_batch_size: int) -> Tuple[Model, ndarray, ndarray]:
    """
    Prepare everything for the attention KT framework.

    :param x_train: the train data.
    :param x_val: the validation data.
    :param teacher: the teacher model.
    :param student: the student model.
    :param evaluation_batch_size: the evaluation batch size
    to be used for the data generated by the attention teacher.
    :return: the attention student and the generated data.
    """
    # Create attention teacher.
    attention_teacher = _teacher_adaptation(teacher)
    # Get attention teacher's outputs.
    y_train = attention_teacher.predict(x_train, evaluation_batch_size, 0)
    y_val = attention_teacher.predict(x_val, evaluation_batch_size, 0)
    # Create attention student.
    attention_student = _student_adaptation(student, input_shape=(attention_teacher.output_shape[1],))

    return attention_student, y_train, y_val
