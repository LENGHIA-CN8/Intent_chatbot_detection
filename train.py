import gym
import tensorflow as tf
from sklearn.metrics import accuracy_score

class SentimentEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, data):
        super(SentimentEnv, self).__init__()
        self.data = data
        self.idx = 0
        self.action_space = gym.spaces.Discrete(3) # Positive, Negative, Neutral
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(300,))
        
    def reset(self):
        self.idx = 0
        return self.data[self.idx]['embedding']
    
    def step(self, action):
        observation = self.data[self.idx]['embedding']
        label = self.data[self.idx]['label']
        reward = 0
        done = False
        
        # Convert action to label
        if action == 0:
            predicted_label = 'Positive'
        elif action == 1:
            predicted_label = 'Negative'
        else:
            predicted_label = 'Neutral'
        
        # Evaluate predicted label against true label
        if predicted_label == label:
            reward = 1
        else:
            reward = -1
        
        self.idx += 1
        if self.idx == len(self.data):
            done = True
            
        return observation, reward, done, {}
    
class SentimentModel(tf.keras.Model):
    def __init__(self):
        super(SentimentModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(3, activation='softmax')
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x
    
def train(env, model, episodes):
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    for i in range(episodes):
        observation = env.reset()
        done = False
        episode_reward = 0
        while not done:
            with tf.GradientTape() as tape:
                action_probs = model(observation[np.newaxis])
                action = tf.random.categorical(action_probs, 1)[0, 0]
                observation_new, reward, done, _ = env.step(action)
                episode_reward += reward
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)([action], action_probs)
                total_loss = loss * reward # Incorporate human feedback
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            observation = observation_new
        print('Episode {}: Total Reward = {}'.format(i+1, episode_reward))
        
def evaluate(env, model, data):
    y_true = []
    y_pred = []
    for i in range(len(data)):
        observation = data[i]['embedding']
        action_probs = model(observation[np.newaxis])
        action = tf.argmax(action_probs, axis=1).numpy()[0]
        y_pred.append(action)
        label = data[i]['label']
        if label == 'Positive':
            y_true.append(0)
        elif label == 'Negative':
            y_true.append(1)
        else:
            y_true.append(2)
    accuracy = accuracy_score(y_true, y_pred)
    print('Accuracy: {:.4f}'.format(accuracy))

# Load data
data = load_data()

# Initialize environment and model
env = SentimentEnv(data)
model = SentimentModel()

train(env, model, episodes=100)

# Evaluate model
evaluate(env, model, data)


# import gym
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression

# # Define the environment class
# class SentimentAnalysisEnv(gym.Env):
#     def __init__(self, data, vocab_size):
#         super().__init__()
#         self.data = data
#         self.vocab_size = vocab_size
#         self.observation_space = gym.spaces.Discrete(vocab_size)
#         self.action_space = gym.spaces.Discrete(2)
#         self.reset()

#     def reset(self):
#         self.current_index = 0
#         self.current_text, self.current_label = self.data[self.current_index]
#         self.vectorizer = CountVectorizer(max_features=self.vocab_size)
#         self.vectorizer.fit([self.current_text])
#         self.current_text_vector = self.vectorizer.transform([self.current_text]).toarray()[0]
#         return self.current_text_vector

#     def step(self, action):
#         reward = 0
#         done = False
#         if action == self.current_label:
#             reward = 1
#         else:
#             reward = -1

#         if self.current_index == len(self.data) - 1:
#             done = True
#         else:
#             self.current_index += 1
#             self.current_text, self.current_label = self.data[self.current_index]
#             self.current_text_vector = self.vectorizer.transform([self.current_text]).toarray()[0]

#         return self.current_text_vector, reward, done, {}

# # Define the agent class
# class SentimentAnalysisAgent:
#     def __init__(self, observation_space, action_space):
#         self.model = LogisticRegression()
#         self.observation_space = observation_space
#         self.action_space = action_space

#     def act(self, observation):
#         return self.model.predict([observation])[0]

#     def update(self, observation, action, reward):
#         self.model.fit([observation], [action], sample_weight=[reward])

# # Define the main function
# def main():
#     # Load the data
#     data = [("I love this movie", 1), ("I hate this movie", 0), ("This movie is great", 1), ("This movie is terrible", 0)]

#     # Define the hyperparameters
#     vocab_size = 1000
#     episodes = 100

#     # Create the environment and agent
#     env = SentimentAnalysisEnv(data, vocab_size)
#     agent = SentimentAnalysisAgent(env.observation_space, env.action_space)

#     # Train the agent using reinforcement learning with human feedback
#     for episode in range(episodes):
#         observation = env.reset()
#         done = False
#         while not done:
#             action = agent.act(observation)
#             observation_next, reward, done, _ = env.step(action)
#             agent.update(observation, action, reward)
#             observation = observation_next

#     # Test the agent on new data
#     test_data = [("This is a fantastic movie", 1), ("This movie is terrible", 0), ("I really enjoyed this film", 1), ("I didn't like this movie", 0)]
#     test_vectorizer = CountVectorizer(max_features=vocab_size)
#     test_texts, test_labels = zip(*test_data)
#     test_vectors = test_vectorizer.fit_transform(test_texts).toarray()
#     test_predictions = agent.model.predict(test_vectors)
#     test_accuracy = np.mean(test_predictions == np.array(test_labels))
#     print("Test accuracy: {:.2f}%".format(test_accuracy * 100))

# if __name__ == "__main__":
#     main()