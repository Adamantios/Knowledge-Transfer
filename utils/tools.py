from tensorflow import zeros_like, stack
from tensorflow.python.keras.engine.base_layer import InputSpec, Layer
from tensorflow.python.keras.layers.merge import _Merge


class Crop(Layer):
    """Layer that crops (or slices) a Tensor on a given dimension from start to end."""

    def __init__(self, dimension, start, end=None, **kwargs):
        self.kernel = None
        self.dimension = dimension
        self.start = start
        self.end = end if end is not None else self.start + 1
        super(Crop, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(Crop, self).build(input_shape)

    def call(self, inputs, **kwargs):
        if self.dimension == 0:
            return inputs[self.start: self.end]
        if self.dimension == 1:
            return inputs[:, self.start: self.end]
        if self.dimension == 2:
            return inputs[:, :, self.start: self.end]
        if self.dimension == 3:
            return inputs[:, :, :, self.start: self.end]
        if self.dimension == 4:
            return inputs[:, :, :, :, self.start: self.end]

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    def get_config(self):
        config = {
            'dimension': self.dimension,
            'start': self.start,
            'end': self.end
        }
        base_config = super(Crop, self).get_config()
        base_config.update(config)
        return base_config


class ZerosLike(Layer):
    """Layer wrapper for tf.zeros_like."""

    def __init__(self, **kwargs):
        super(ZerosLike, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(ZerosLike, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return zeros_like(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.output_dim

    def get_config(self):
        base_config = super(ZerosLike, self).get_config()
        return base_config


class Stack(_Merge):
    """Merge Layer wrapper for tf.stack."""

    def __init__(self, axis=0, **kwargs):
        self.axis = axis
        super(Stack, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Stack, self).build(input_shape)

    def _merge_function(self, inputs):
        return stack(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('A `Stack` layer should be called on a list of inputs.')
        input_shapes = input_shape
        output_shape = list(input_shapes[0])
        for shape in input_shapes[1:]:
            if output_shape[self.axis] is None or shape[self.axis] is None:
                output_shape[self.axis] = None
                break
            output_shape[self.axis] += shape[self.axis]
        return tuple(output_shape)

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Stack, self).get_config()
        base_config.update(config)
        return base_config
