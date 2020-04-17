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
                            initializer='uniform',
                            regularizer="l2"
                            )
        self.u_z = self.add_weight(
                            name="u_update",
                            shape=(self.units, self.units),
                            initializer='uniform',
                            regularizer="l2"
                            )
        self.b_z = self.add_weight(
                            name="b_update",
                            shape=(self.units,),
                            initializer='zeros',
                            regularizer="l2"                       
                            )
        # Reset gate
        self.w_r = self.add_weight(
                            name="w_reset",
                            shape=(self.input_dim, self.units),
                            initializer='uniform',
                            regularizer="l2"
                            )
        self.u_r = self.add_weight(
                            name="u_reset",
                            shape=(self.units, self.units),
                            initializer='uniform',
                            regularizer="l2"
                            )
        self.b_r = self.add_weight(
                            name="b_reset",
                            shape=(self.units,),
                            initializer='zeros',
                            regularizer="l2"                        
                            )
        # Memory content
        self.w_h = self.add_weight(
                            name="w_memory",
                            shape=(self.input_dim, self.units),
                            initializer='uniform',
                            regularizer="l2"
                            )
        self.u_h = self.add_weight(
                            name="u_memory",
                            shape=(self.units, self.units),
                            initializer='uniform',
                            regularizer="l2"                            
                            )
        self.b_h = self.add_weight(
                            name="b_memory",
                            shape=(self.units,),
                            initializer='zeros',
                            regularizer="l2"                        
                            )
            
    def call(self, inputs, hidden_states):
        # Mask the hidden state to reset it at timesteps with finished environments
        input, mask = tf.split(inputs, 2, axis=1)
        out_mask = tf.matmul(mask, tf.zeros((mask.shape[-1], self.units)))
        h_masked = hidden_states[0] * out_mask

        # Compute update and reset gates
        z_t = tf.nn.sigmoid(tf.matmul(input, self.w_z) + tf.matmul(h_masked, self.u_z) + self.b_z)
        r_t = tf.nn.sigmoid(tf.matmul(input, self.w_r) + tf.matmul(h_masked, self.u_r) + self.b_r)
        
        # Compute current hidden state (memory content)
        h_t = tf.nn.tanh(tf.matmul(input, self.w_h) + tf.matmul((r_t * h_masked), self.u_h) + self.b_h)
        h_t_seq = (z_t * h_t) + ((1 - z_t) * h_masked)

        h_t_forward = tf.concat((h_t_seq, out_mask), axis=1)
        return h_t_forward, [h_t_seq]