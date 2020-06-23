'''We will model a simple weather system and try to predict the temperature on each day given the following information.

1. Cold days are encoded by a 0 and hot days are encoded by a 1.
2. The first day in our sequence has an 80% chance of being cold.
3. A cold day has a 30% chance of being followed by a hot day.
4. A hot day has a 20% chance of being followed by a cold day.
5. On each day the temperature is normally distributed with mean and standard deviation 0 and 5 on a cold day and mean
and standard deviation 15 and 10 on a hot day.'''

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

init_dist = tfd.Categorical(probs=[0.8, 0.2]) # Refer to point 2
trans_dist = tfd.Categorical(probs=[[0.5, 0.5],
                                    [0.2, 0.8]]) # Refer to point 3 and 4
obs_dist = tfd.Normal(loc=[0., 15.], scale=[5., 10.]) # Refer to point 5
steps = 7

model = tfd.HiddenMarkovModel(initial_distribution=init_dist,
                              transition_distribution=trans_dist,
                              observation_distribution=obs_dist,
                              num_steps=steps)

mean = model.mean()

# due to the way TensorFlow works on a lower level we need to evaluate part of the graph
# from within a session to see the value of this tensor

# in the new version of tensorflow we need to use tf.compat.v1.Session() rather than just tf.Session()
with tf.compat.v1.Session() as sess:
  print(mean.numpy())
