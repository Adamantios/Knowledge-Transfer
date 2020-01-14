from typing import Tuple

from numpy import argmax, concatenate
from numpy.core.multiarray import ndarray
from tensorflow import zeros_like, stack
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Concatenate, Activation, Dense, Multiply


def _pyramid_ensemble_adaptation(teacher: Model) -> Tuple[Model, int]:
    """
    Adapt pyramid ensemble by changing its output to contain each of its submodels outputs.

    :param teacher: the pyramid ensemble.
    :return: the attention pyramid ensemble and the submodels number.
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
    outputs = stack([output1, output2, output3], name='concatenated_submodels_outputs')

    # Create attention teacher.
    attention_teacher = Model(teacher.input, outputs, name='attention_' + teacher.name)
    return attention_teacher, 3


def _complicated_ensemble_adaptation(teacher: Model) -> Tuple[Model, int]:
    """
    Adapt complicated ensemble by changing its output to contain each of its submodels outputs.

    :param teacher: the complicated ensemble.
    :return: the attention complicated ensemble.
    """
    # Get each submodel's outputs.
    output1 = teacher.layers[-7].output
    output2 = teacher.layers[-6].output
    output3 = teacher.layers[-5].output
    output4 = teacher.layers[-4].output
    output5 = teacher.layers[-3].output

    # Append zeros to the model outputs which do not predict all the classes.
    output1_fixed = Concatenate(name='output_1_fixed')(
        [output1, zeros_like(output2), zeros_like(output3), zeros_like(output4), zeros_like(output5)]
    )
    output2_fixed = Concatenate(name='output_2_fixed')(
        [zeros_like(output1), output2, zeros_like(output3), zeros_like(output4), zeros_like(output5)]
    )
    output3_fixed = Concatenate(name='output_3_fixed')(
        [zeros_like(output1), zeros_like(output2), output3, zeros_like(output4), zeros_like(output5)]
    )
    output4_fixed = Concatenate(name='output_4_fixed')(
        [zeros_like(output1), zeros_like(output2), zeros_like(output3), output4, zeros_like(output5)]
    )
    output5_fixed = Concatenate(name='output_5_fixed')(
        [zeros_like(output1), zeros_like(output2), zeros_like(output3), zeros_like(output4), output5]
    )
    # Add activations to the outputs.
    output1_fixed = Activation('softmax', name='softmax1')(output1_fixed)
    output2_fixed = Activation('softmax', name='softmax2')(output2_fixed)
    output3_fixed = Activation('softmax', name='softmax3')(output3_fixed)
    output4_fixed = Activation('softmax', name='softmax4')(output4_fixed)
    output5_fixed = Activation('softmax', name='softmax5')(output5_fixed)
    # Stack submodels outputs.
    outputs = stack([output1_fixed, output2_fixed, output3_fixed, output4_fixed, output5_fixed],
                    axis=1, name='concatenated_submodels_outputs')

    # Create attention teacher.
    attention_teacher = Model(teacher.input, outputs, name='attention_' + teacher.name)
    return attention_teacher, 5


def _ensemble_adaptation(teacher: Model) -> Tuple[Model, int]:
    """
    Adapt an averaged predictions ensemble by changing its output to contain each of its submodels outputs.

    :param teacher: the ensemble.
    :return: the attention ensemble.
    """
    # Calculate the number of submodels.
    submodels_num = len(teacher.layers[1:-1])

    # Concatenate submodels outputs.
    outputs = stack([teacher.layers[i + 1](teacher.input) for i in range(submodels_num)],
                    name='concatenated_submodels_outputs')

    # Create attention teacher.
    attention_teacher = Model(teacher.input, outputs, name='attention_' + teacher.name)
    return attention_teacher, submodels_num


def _teacher_adaptation(teacher: Model) -> Tuple[Model, int]:
    """
    Adapt teacher by changing its output to contain each of the submodels outputs.

    :param teacher: the teacher to adapt.
    :return: the attention teacher.
    """
    if 'pyramid_ensemble' in teacher.name:
        attention_teacher, submodels_num = _pyramid_ensemble_adaptation(teacher)
    elif 'complicated_ensemble' in teacher.name:
        attention_teacher, submodels_num = _complicated_ensemble_adaptation(teacher)
    elif teacher.name == 'ensemble':
        attention_teacher, submodels_num = _ensemble_adaptation(teacher)
    else:
        raise ValueError('Unknown teacher model has been given.')

    return attention_teacher, submodels_num


def _student_adaptation(student: Model, submodels_num, input_shape: tuple) -> Model:
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
    attention_vector = Dense(submodels_num, activation='softmax', name='attention_vector')(attention_inputs)
    # Choose a teacher.
    chosen_teacher = attention_inputs[:, argmax(attention_vector)]
    # Multiply teacher values with student's outputs.
    outputs = Multiply(name='attention_weighted_predictions')([student(student_input), chosen_teacher])
    # Add a softmax.
    outputs = Activation('softmax', name='attention_weighted_predictions_softmax')(outputs)

    # Create and return attention student.
    attention_student = Model([student_input, attention_inputs], outputs, name='attention_' + student.name)
    return attention_student


def attention_framework_adaptation(x_train: ndarray, teacher: Model, student: Model,
                                   evaluation_batch_size: int) -> Tuple[Model, ndarray]:
    """
    Prepare everything for the attention KT framework.

    :param x_train: the train data.
    :param teacher: the teacher model.
    :param student: the student model.
    :param evaluation_batch_size: the evaluation batch size
    to be used for the data generated by the attention teacher.
    :return: the attention student and the generated data.
    """
    # Create attention teacher.
    attention_teacher, submodels_num = _teacher_adaptation(teacher)
    # Get attention teacher's outputs.
    y_train = attention_teacher.predict(x_train, evaluation_batch_size, 0)
    # Concatenate outputs.
    y_train = concatenate([y_train[:, i] for i in range(y_train.shape[1])], axis=1)
    # Create attention student.
    attention_student = _student_adaptation(student, submodels_num,
                                            input_shape=(attention_teacher.output_shape[2] * submodels_num,))

    return attention_student, y_train
