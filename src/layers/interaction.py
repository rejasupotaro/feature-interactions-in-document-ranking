import itertools

import tensorflow as tf
from tensorflow.keras import layers


class WeightedQueryFieldInteraction(tf.keras.layers.Layer):
    def __init__(self, num_fields, **kwargs):
        self.num_fields = num_fields
        super(WeightedQueryFieldInteraction, self).__init__(**kwargs)

    def build(self, input_shape):
        w_init = tf.constant_initializer(value=0)
        self.field_weights = tf.Variable(
            initial_value=w_init(shape=(self.num_fields - 1), dtype=tf.float32),
        )
        super(WeightedQueryFieldInteraction, self).build(input_shape)

    def call(self, inputs, **kwargs):
        dim = inputs.shape[1] // self.num_fields
        interactions = []
        query = inputs[:, 0:dim]
        for i in range(self.num_fields - 1):
            field = inputs[:, i * dim:(i + 1) * dim]
            interaction = layers.Dot(axes=1)([query, field])
            interaction = tf.math.scalar_mul(self.field_weights[i], interaction)
            interactions.append(interaction)
        interactions = layers.Add()(interactions)
        return interactions

    def compute_output_shape(self, input_shape):
        return None, 1

    def get_config(self):
        config = super(WeightedQueryFieldInteraction, self).get_config().copy()
        config.update({
            'num_fields': self.num_fields,
        })
        return config


class WeightedFeatureInteraction(tf.keras.layers.Layer):
    def __init__(self, num_fields, **kwargs):
        self.num_fields = num_fields
        super(WeightedFeatureInteraction, self).__init__(**kwargs)

    def build(self, input_shape):
        w_init = tf.constant_initializer(value=0)
        self.field_weights = tf.Variable(
            initial_value=w_init(shape=(self.num_fields, self.num_fields), dtype=tf.float32),
        )
        super(WeightedFeatureInteraction, self).build(input_shape)

    def call(self, inputs, **kwargs):
        dim = inputs.shape[1] // self.num_fields
        interactions = []
        for i, j in itertools.combinations(range(self.num_fields), 2):
            interaction = layers.Dot(axes=1)([inputs[:, i * dim:(i + 1) * dim], inputs[:, j * dim:(j + 1) * dim]])
            interaction = tf.math.scalar_mul(self.field_weights[i, j], interaction)
            interactions.append(interaction)
        interactions = layers.Add()(interactions)
        return interactions

    def compute_output_shape(self, input_shape):
        return None, 1

    def get_config(self):
        config = super(WeightedFeatureInteraction, self).get_config().copy()
        config.update({
            'num_fields': self.num_fields,
        })
        return config


class WeightedSelectedFeatureInteraction(tf.keras.layers.Layer):
    def __init__(self, num_fields, **kwargs):
        self.num_fields = num_fields
        super(WeightedSelectedFeatureInteraction, self).__init__(**kwargs)

    def build(self, input_shape):
        w_init = tf.constant_initializer(value=0)
        self.field_weights = tf.Variable(
            initial_value=w_init(shape=(self.num_fields - 1), dtype=tf.float32),
        )
        super(WeightedSelectedFeatureInteraction, self).build(input_shape)

    def interaction(self, f1, f2, i):
        interaction = layers.Dot(axes=1)([f1, f2])
        return tf.math.scalar_mul(self.field_weights[i], interaction)

    def call(self, inputs, **kwargs):
        dim = inputs.shape[1] // self.num_fields
        query = inputs[:, 0:dim]
        title = inputs[:, 1 * dim:2 * dim]
        ingredients = inputs[:, 2 * dim:3 * dim]
        description = inputs[:, 3 * dim:4 * dim]
        country = inputs[:, 4 * dim:5 * dim]
        image = inputs[:, 5 * dim:6 * dim]

        interactions = [
            self.interaction(title, country, 0),
            self.interaction(title, image, 0),
            self.interaction(title, ingredients, 0),
            self.interaction(query, title, 0),
            self.interaction(ingredients, country, 0),
        ]
        interactions = layers.Add()(interactions)
        return interactions

    def compute_output_shape(self, input_shape):
        return None, 1

    def get_config(self):
        config = super(WeightedSelectedFeatureInteraction, self).get_config().copy()
        config.update({
            'num_fields': self.num_fields,
        })
        return config
