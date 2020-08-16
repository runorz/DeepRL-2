# The script refer to https://keras.io/examples/rl/ddpg_pendulum/
import tensorflow as tf
import numpy as np
import pandas as pd
import os
# model = tf.keras.Sequential([
#     tf.keras.layers.InputLayer(input_shape=(5,)),
#     tf.keras.layers.Dense(units=64, activation='relu'),
#     tf.keras.layers.Dense(units=20, activation='sigmoid')
# ])

# params
num_states = 3
num_actions = 8  # 2N

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.buffer_counter += 1

    def learn(self):
        record_range = min(self.buffer_counter, self.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch)
            y = reward_batch + gamma * target_critic([next_state_batch, target_actions])
            critic_value = critic_model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch)
            critic_value = critic_model([state_batch, actions])
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

def update_target(tau):
    new_weights = []
    target_variables = target_critic.weights
    for i, variable in enumerate(critic_model.weights):
        new_weights.append(variable * tau + target_variables[i] * (1 - tau))
    target_critic.set_weights(new_weights)

    new_weights = []
    target_variables = target_actor.weights
    for i, variable in enumerate(actor_model.weights):
        new_weights.append(variable * tau + target_variables[i] * (1 - tau))
    target_actor.set_weights(new_weights)



def get_actor():
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = tf.keras.layers.Input(shape=(num_states,))
    out = tf.keras.layers.Dense(512, activation='relu')(inputs)
    out = tf.keras.layers.BatchNormalization()(out)
    # out = tf.keras.layers.Dense(512, activation='relu')(out)
    # out = tf.keras.layers.BatchNormalization()(out)
    # outputs = tf.keras.layers.Dense(4, activation='sigmoid', kernel_initializer=last_init)(out)
    outputs = tf.keras.layers.Dense(num_actions, activation='tanh', kernel_initializer=last_init)(out)
    # outputs = outputs * max_action_force
    # outputs = tf.keras.layers.Dense(4)(out)
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    state_input = tf.keras.layers.Input(shape=(num_states))
    # state_out = tf.keras.layers.Dense(32, activation='relu')(state_input)
    # state_out = tf.keras.layers.BatchNormalization()(state_out)
    # state_out = tf.keras.layers.Dense(32, activation='relu')(state_out)
    # state_out = tf.keras.layers.BatchNormalization()(state_input)

    action_input = tf.keras.layers.Input(shape=(num_actions))
    # action_out = tf.keras.layers.Dense(32, activation='relu')(action_input)
    # action_out = tf.keras.layers.BatchNormalization()(action_out)

    concat = tf.keras.layers.Concatenate()([state_input, action_input])

    out = tf.keras.layers.Dense(512, activation='relu')(concat)
    out = tf.keras.layers.BatchNormalization()(out)
    # out = tf.keras.layers.Dense(512, activation='relu')(out)
    # out = tf.keras.layers.BatchNormalization()(out)
    outputs = tf.keras.layers.Dense(1)(out)

    model = tf.keras.Model([state_input, action_input], outputs)

    return model

def process_file(f, val):
    pre_state = f['pre_state']
    action = f['action']
    reward = f['reward']
    state = f['state']

    pre_state = [e.split() for e in pre_state]
    pre_state = [[float(i) for i in e] for e in pre_state]

    action = [e.split() for e in action]
    action = [[float(i) for i in e] for e in action]

    state = [e.split() for e in state]
    state = [[float(i) for i in e] for e in state]

    print(f"average reward: {np.mean(reward)} ")
    #avg_records.append(np.mean(reward))
    if val == 0:
        with open('records.csv', 'a') as fd:
            fd.write(f'{np.mean(reward)}\n')
    elif val == 1:
        with open('testing_records.csv', 'a') as fd:
            fd.write(f'{np.mean(reward)}\n')
    elif val == 2:
        with open('testing_target_records.csv', 'a') as fd:
            fd.write(f'{np.mean(reward)}\n')

    for i in range(len(pre_state)):
        buffer.record((pre_state[i], action[i], reward[i], state[i]))


actor_model = get_actor()
critic_model = get_critic()
target_actor = get_actor()
target_critic = get_critic()

gamma = 0.99

critic_lr = 0.0001
actor_lr = 0.0001

tau = 0.005

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

buffer = Buffer()

# print(actor_model(tf.constant([[0.4, 0.2, 0.3, 0.8, 0.2]])))
# print(critic_model([tf.constant([[0.4, 0.2, 0.3, 0.8, 0.2]]), tf.constant([[0.2, 0.2, 0.1, 0.3]])]))
# actor_model.summary()
# critic_model.summary()

total_episodes = 10000

avg_records = []


for ep in range(total_episodes):
    print(f"In current episode:{ep}")
    val = 0
    if ep % 10 == 0:
        val = 2
        target_actor.save(f'model/target_action_model.h5', include_optimizer=False)
        os.system(f"python3 model/convert_model.py model/target_action_model.h5 model/target_action_model.json")
        os.system(f'./build/testing_target')
        f = pd.read_csv(f'record/record.csv')
        process_file(f, val)
        val = 1
        actor_model.save(f'model/action_model.h5', include_optimizer=False)
        os.system(f"python3 model/convert_model.py model/action_model.h5 model/action_model.json")
        os.system(f'./build/testing')
        f = pd.read_csv(f'record/record.csv')
        process_file(f, val)
    else:
        actor_model.save(f'model/action_model.h5', include_optimizer=False)
        os.system(f"python3 model/convert_model.py model/action_model.h5 model/action_model.json")
        os.system(f"./build/training")
        f = pd.read_csv(f'record/record.csv')
        process_file(f, val)

    for _ in range(10):
        buffer.learn()
        update_target(tau)




# actor_model.save('model/action_model.h5', include_optimizer=False)
# os.system("python3 model/convert_model.py model/action_model.h5 model/action_model.json")

# pd.DataFrame(avg_records).to_csv('avg_records.csv')
print("Hello")
