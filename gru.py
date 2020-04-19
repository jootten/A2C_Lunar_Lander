import tensorflow as tf
# Custom GRUCell

class GRUCell(tf.keras.layers.Layer):

    def __init__(self, input_dim, units):
        super(GRUCell, self).__init__()
        self.input_dim = input_dim
        self.units = units
        self.itteration = 0
        # TF needs this.
        self.state_size = units
    
    def build(self, input_shape):
        # Update gate
        self.w_z = self.add_weight(
                            name="w_update",
                            shape=(self.input_dim, self.units),
                            initializer='random_normal',
                            regularizer="l2"
                            )
        self.u_z = self.add_weight(
                            name="u_update",
                            shape=(self.units, self.units),
                            initializer='random_normal',
                            regularizer="l2"
                            )
        self.b_z = self.add_weight(
                            name="b_update",
                            shape=(self.units,),
                            initializer='zeros',
                            regularizer=None                       
                            )
        # Reset gate
        self.w_r = self.add_weight(
                            name="w_reset",
                            shape=(self.input_dim, self.units),
                            initializer='random_normal',
                            regularizer="l2"
                            )
        self.u_r = self.add_weight(
                            name="u_reset",
                            shape=(self.units, self.units),
                            initializer='random_normal',
                            regularizer=None
                            )
        self.b_r = self.add_weight(
                            name="b_reset",
                            shape=(self.units,),
                            initializer='zeros',
                            regularizer=None                       
                            )
        # Memory content
        self.w_h = self.add_weight(
                            name="w_memory",
                            shape=(self.input_dim, self.units),
                            initializer='random_normal',
                            regularizer="l2"
                            )
        self.u_h = self.add_weight(
                            name="u_memory",
                            shape=(self.units, self.units),
                            initializer='random_normal',
                            regularizer="l2"                            
                            )
        self.b_h = self.add_weight(
                            name="b_memory",
                            shape=(self.units,),
                            initializer='zeros',
                            regularizer=None                        
                            )
            
    def call(self, inputs, hidden_states):
        # Mask the hidden state to reset it at timesteps with finished environments
        input, mask = tf.split(inputs, 2, axis=1)
        mask = tf.matmul(mask, tf.ones((mask.shape[-1], self.units))) / self.input_dim
        h_masked = hidden_states[0] * mask

        # Compute update and reset gates
        z_t = tf.nn.sigmoid(tf.matmul(input, self.w_z) + tf.matmul(h_masked, self.u_z) + self.b_z)
        r_t = tf.nn.sigmoid(tf.matmul(input, self.w_r) + tf.matmul(h_masked, self.u_r) + self.b_r)
        
        # Compute current hidden state (memory content)
        h_t = tf.nn.tanh(tf.matmul(input, self.w_h) + tf.matmul((r_t * h_masked), self.u_h) + self.b_h)
        h_t = (z_t * h_t) + ((1 - z_t) * h_masked)

        h_t_forward = tf.concat((h_t, mask), axis=1)
        return h_t_forward, [h_t]