from typing import Tuple

from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Concatenate, Activation

from utils.tools import ZerosLike, Stack


def _pyramid_ensemble_adaptation(teacher: Model) -> Tuple[Model, int]:
    """
    Adapt pyramid ensemble by changing its output to contain each of its submodels outputs.

    :param teacher: the pyramid ensemble.
    :return: the selective_learning pyramid ensemble and the submodels number.
    """
    # Get each submodel's outputs.
    output1 = teacher.get_layer('submodel_strong_output').output
    weak_1_output = teacher.get_layer('submodel_weak_1_output').output
    weak_2_output = teacher.get_layer('submodel_weak_2_output').output
    # Create zeros.
    weak_1_zeros = ZerosLike(name='weak_1_zeros')(weak_1_output)
    weak_2_zeros = ZerosLike(name='weak_2_zeros')(weak_2_output)
    # Append zeros to the model outputs which do not predict all the classes.
    output2 = Concatenate(name='submodel_weak_1_output_fixed')([weak_1_output, weak_1_zeros])
    output3 = Concatenate(name='submodel_weak_2_output_fixed')([weak_2_zeros, weak_2_output])
    # Add activations to the outputs.
    output1 = Activation('softmax', name='softmax1')(output1)
    output2 = Activation('softmax', name='softmax2')(output2)
    output3 = Activation('softmax', name='softmax3')(output3)
    # Stack submodels outputs.
    outputs = Stack(axis=1, name='stacked_submodels_outputs')([output1, output2, output3])

    # Create selective_learning teacher.
    selective_learning_teacher = Model(teacher.input, outputs, name='selective_learning_' + teacher.name)
    return selective_learning_teacher, 3


def _complicated_ensemble_adaptation(teacher: Model) -> Tuple[Model, int]:
    """
    Adapt complicated ensemble by changing its output to contain each of its submodels outputs.

    :param teacher: the complicated ensemble.
    :return: the selective_learning complicated ensemble and the number of submodels.
    """
    # Get each submodel's outputs.
    output1 = teacher.layers[-7].output
    output2 = teacher.layers[-6].output
    output3 = teacher.layers[-5].output
    output4 = teacher.layers[-4].output
    output5 = teacher.layers[-3].output

    # Create zeros.
    zeros1 = ZerosLike(name='zeros1')(output1)
    zeros2 = ZerosLike(name='zeros2')(output2)
    zeros3 = ZerosLike(name='zeros3')(output3)
    zeros4 = ZerosLike(name='zeros4')(output4)
    zeros5 = ZerosLike(name='zeros5')(output5)

    # Append zeros to the model outputs which do not predict all the classes.
    output1_fixed = Concatenate(name='output_1_fixed')([output1, zeros2, zeros3, zeros4, zeros5])
    output2_fixed = Concatenate(name='output_2_fixed')([zeros1, output2, zeros3, zeros4, zeros5])
    output3_fixed = Concatenate(name='output_3_fixed')([zeros1, zeros2, output3, zeros4, zeros5])
    output4_fixed = Concatenate(name='output_4_fixed')([zeros1, zeros2, zeros3, output4, zeros5])
    output5_fixed = Concatenate(name='output_5_fixed')([zeros1, zeros2, zeros3, zeros4, output5])
    # Add activations to the outputs.
    output1_fixed = Activation('softmax', name='softmax1')(output1_fixed)
    output2_fixed = Activation('softmax', name='softmax2')(output2_fixed)
    output3_fixed = Activation('softmax', name='softmax3')(output3_fixed)
    output4_fixed = Activation('softmax', name='softmax4')(output4_fixed)
    output5_fixed = Activation('softmax', name='softmax5')(output5_fixed)
    # Stack submodels outputs.
    outputs = Stack(axis=1, name='stacked_submodels_outputs')(
        [output1_fixed, output2_fixed, output3_fixed, output4_fixed, output5_fixed]
    )

    # Create selective_learning teacher.
    selective_learning_teacher = Model(teacher.input, outputs, name='selective_learning_' + teacher.name)
    return selective_learning_teacher, 5


def _ensemble_adaptation(teacher: Model) -> Tuple[Model, int]:
    """
    Adapt an averaged predictions ensemble by changing its output to contain each of its submodels outputs.

    :param teacher: the ensemble.
    :return: the selective_learning ensemble and the number of submodels.
    """
    # Calculate the number of submodels.
    submodels_num = len(teacher.layers[1:-1])

    # Stack submodels outputs.
    outputs = Stack(axis=1, name='stacked_submodels_outputs')(
        [teacher.layers[i + 1](teacher.input) for i in range(submodels_num)]
    )

    # Create selective_learning teacher.
    selective_learning_teacher = Model(teacher.input, outputs, name='selective_learning_' + teacher.name)
    return selective_learning_teacher, submodels_num


def selective_learning_teacher_adaptation(teacher: Model) -> Tuple[Model, int]:
    """
    Adapt teacher by changing its output to contain each of the submodels outputs.

    :param teacher: the teacher to adapt.
    :return: the selective_learning teacher and the number of submodels.
    """
    if 'pyramid_ensemble' in teacher.name:
        selective_learning_teacher, submodels_num = _pyramid_ensemble_adaptation(teacher)
    elif 'complicated_ensemble' in teacher.name:
        selective_learning_teacher, submodels_num = _complicated_ensemble_adaptation(teacher)
    elif teacher.name == 'ensemble':
        selective_learning_teacher, submodels_num = _ensemble_adaptation(teacher)
    else:
        raise ValueError('Unknown teacher model has been given.')

    return selective_learning_teacher, submodels_num


def selective_learning_student_adaptation(student: Model, n_models: int) -> Model:
    """
    Adapt student by changing its outputs to predict the classes for every sub-teacher.

    :param student: the student to adapt.
    :param n_models: the number of teacher submodels.
    :return: the selective_learning student.
    """
    outputs = []
    for input_idx in range(n_models):
        # Initialize the teacher submodel.
        submodel = Model(student.input, student.output, name='student_{}'.format(input_idx))
        # Generate outputs.
        outputs.append(submodel(submodel.input))

    # Create selective_learning student.
    selective_learning_student = Model(student.input, outputs, name='selective_learning_{}'.format(student.name))

    return selective_learning_student


def selective_learning_student_rewind(student: Model, **compilation_data) -> Model:
    """
    Reverts the selective_learning framework transformation from the student.

    :param student: the student to be rewinded.
    :return: the rewinded student.
    """
    # Just get any submodel's output, because after training the selective_learning student,
    # all its submodels outputs are going to have the same weights.
    submodel = student.layers[1]
    outputs = submodel.call(student.input)
    rewinded_student = Model(student.input, outputs, name=student.name.replace('selective_learning_', ''))

    # Recompile model.
    rewinded_student.compile(**compilation_data)

    return rewinded_student
