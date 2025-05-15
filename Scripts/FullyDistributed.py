import gym
from gym import spaces
#import veins_gym
import numpy as np
import pickle
import random
import nashpy as nash
from collections import defaultdict
import argparse
import time

def encoder(vector, ranges):
    scalar = 0
    multiplier = 1
    for v, r in zip(vector, ranges):
        scalar += v * multiplier
        multiplier *= r
    return scalar

def decoder(scalar, ranges):
    vector = []
    for r in ranges:
        vector.append(scalar % r)
        scalar //= r
    return vector

class MockRSUEnv(gym.Env):
    def __init__(self, action_ranges, state_ranges, episode_length=15):
        super(MockRSUEnv, self).__init__()
        self.episode_length = episode_length
        self.step_count = 0
        self.action_ranges = action_ranges
        # all agents state (queries, caching state)
        self.state_ranges = state_ranges
        self.total_action_size = np.prod(self.action_ranges)
        self.total_state_size = np.prod(self.state_ranges)
        self.observation_space = spaces.Discrete(self.total_state_size)
        self.action_space = spaces.Discrete(self.total_action_size)
        self.state = self.observation_space.sample()

    def reset(self):
        self.step_count = 0
        self.state = self.observation_space.sample()
        return self.state

    def step(self, encoded_action):
        self.step_count += 1
        done = self.step_count >= self.episode_length
        # Decode action
        action_vector = decoder(encoded_action, self.action_ranges)
        reward = 2.0
        reward = reward / len(action_vector)  # normalize
        self.state = self.observation_space.sample()
        return self.state, reward, done, {}

class QLearningAgent:
    def __init__(self, agent_id, state_size, action_size, is_primary, alpha=0.1, gamma=0.95, epsilon=0.2):
        self.agent_id = agent_id
        self.action_size = action_size
        self.is_primary = is_primary
        #self.q_table = defaultdict(lambda: np.zeros(action_size))
        self.q_table = np.zeros((state_size, action_size))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    def load(self, save_dir, episode):
        print("Agent {} loading q table at episode {} ...".format(self.agent_id, episode))
        with open(save_dir+ "q_table_"+ str(self.agent_id)+ ".pkl", "rb") as fd:
            self.q_table = pickle.load(fd)
    def save(self, save_dir, episode):
        print("[Python] Agent {} saving q table at episode {} ...".format(self.agent_id, episode))
        with open(save_dir+ "q_table_"+ str(self.agent_id)+ ".pkl", "wb") as fd:
            pickle.dump(self.q_table, fd)
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_table[state])

    def update_q(self, state, action, reward, next_state, equilibrium_val):
        old_val = self.q_table[state][action]
        self.q_table[state][action] = (1 - self.alpha) * old_val + self.alpha * (reward + self.gamma * equilibrium_val)


def main(episode_start=0, nb_vehicules = 6, nb_rsus = 3, nb_content = 3):
    veins_gym_env = False
    save_dir = "./checkpoint/fullydistributed/"
    I = nb_rsus  # number of RSUs (leaders)
    F = nb_content  # content categories
    N = nb_rsus + nb_vehicules  # total agents (vehicles + RSUs)
    H = np.zeros((N,F)) # N F, for each agent caching state varies from 0 to 1
    C = np.zeros((N,F)) # for each vehicule, values range from 0 to F-1; the matrix form make it range between 0 and 1. 
    
    # agents (RSU) actions; [cache or no cache, which content to replace]
    action_ranges_RSU = np.asarray([2,F])
    # agents (vehicules) actions [v2v or v2r, cache or no cache, which content to replace]
    action_ranges_vehicules = np.asarray([2,2,F])
    
    # all agents (RSU and vehicules) state (queries, caching state)
    state_ranges_RSU = np.concatenate((np.full((F), 2), np.full((N*F), 2)))
    state_ranges_vehicules = np.concatenate((np.full((1), F), np.full((N*F), 2))) # its requests and the caching state

    all_action_ranges = np.concatenate((np.full((N - I), 2), np.full((N), 2), np.full((N), F)))
    all_state_ranges = np.concatenate((np.full((N - I), F), np.full((N*F), 2)))
    action_size_RSU = np.prod(action_ranges_RSU)
    action_size_vehicules = np.prod(action_ranges_vehicules)
    state_size_RSU = np.prod(state_ranges_RSU)
    state_size_vehicules = np.prod(state_ranges_vehicules)
    print("[Python] RSU : Agents state size {}, Agents action size {}".format(state_size_RSU, action_size_RSU))
    print("[Python] Vehicules : Agents state size {}, Agents action size {}".format(state_size_vehicules, action_size_vehicules))
    
    # the env accepts all (vehicules + rsu) state and action sizes
    if veins_gym_env:
        gym.register(
            id="veins-v1",
            entry_point = "veins_gym:VeinsEnv",
            kwargs={
            "scenario_dir": "cities",
            "print_veins_stdout": True,
            "timeout": 10.0    
            },
            )
        env = gym.make("veins-v1")
    else:
        env = MockRSUEnv(all_action_ranges, all_state_ranges)
    
    print("[Python] Fully distributed MARL Training Start...")
    agents = {}
    START_FROM_EPISODE = episode_start
    for i in range(N):
        if i < I:
            agents[i] = QLearningAgent(agent_id = i, state_size= state_size_RSU, action_size = action_size_RSU, is_primary = True)
        else:
            agents[i] = QLearningAgent(agent_id = i, state_size= state_size_vehicules, action_size = action_size_vehicules, is_primary = False)
    print("[Python] Number of agents (N) {}".format(len(agents)))
    episode_rewards = []
    episode_flops = []
    episode_hits = []
    episode_misses = []

    if START_FROM_EPISODE > 0:
        print("[Python] Loading ... {START_FROM_EPISODE}")
        for i in range(N):
            agents[i].load(save_dir, START_FROM_EPISODE)
        with open(save_dir + "metrics_"+str(START_FROM_EPISODE)+".pkl") as fd:
            episode_rewards, episode_flops, episode_hits, episode_misses = pickle.load(fd)    
    else:
        print("[Python] Training from scratch ... ")
    
    for episode in range(START_FROM_EPISODE, 10):
        observation = env.reset() # global observation (vehicules + rsu)
        done = False
        episode_reward = 0.0
        episode_flop = 0.0
        episode_hit = 0.0
        episode_miss = 0.0
        while not done:
            start_time = time.time()
            # we obtained global observation from the environment
            decoded_obs = decoder(observation, all_state_ranges)
            # global action to send to environment
            action = np.concatenate((np.full((N - I), 0), np.full((N), 0), np.full((N), 0)))
            actions = {}
            p_states = {}
            C = np.zeros((N,F))
            for i in range(N):
                if i < I: # RSU
                    decoded_p_observation = np.concatenate((np.full((F), 0), np.full((N*F), 0)))
                    for k in range(N*F):
                        decoded_p_observation[F+ k] = decoded_obs[N-I + k]
                        ii = k // F
                        f = k % F
                        H[ii][f] = decoded_obs[N-I + k]
                    for k in range(N-I):
                        f = decoded_obs[k]
                        C[i][f] = 1
                        decoded_p_observation[f] = 1
                    p_states[i] = encoder(decoded_p_observation, state_ranges_RSU)
                    actions[i] = agents[i].choose_action(p_states[i])
                    dec_action = decoder(actions[i], action_ranges_RSU)
                    action[N-I+i] = dec_action[0] # took caching decision.
                    action[2*N-I+i] = dec_action[1] # took the replacement decision.
                else: # Vehicules
                    decoded_p_observation = np.concatenate((np.full((1), 0), np.full((N*F), 0)))
                    f = decoded_obs[i-I]
                    decoded_p_observation[0] = f
                    C[i][f] = 1
                    for k in range(N*F):
                        decoded_p_observation[1+ k] = decoded_obs[N-I + k]
                        ii = k // F
                        f = k % F
                        H[ii][f] = decoded_obs[N-I + k]
                    p_states[i] = encoder(decoded_p_observation, state_ranges_vehicules)
                    actions[i] = agents[i].choose_action(p_states[i])
                    dec_action = decoder(actions[i], action_ranges_vehicules)
                    action[i-I] = dec_action[0]
                    action[N-I+i] = dec_action[1]
                    action[2*N-I+i] = dec_action[2]
        
            # combine back into scalar
            encoded_action = encoder(action, all_action_ranges)
            # Step environment
            obs_next, reward, done, info = env.step(encoded_action)
            decoded_obs_next = decoder(obs_next, all_state_ranges)
            p_states_next = {}
            C = np.zeros((N,F))
            for i in range(N):
                if i < I: #RSU
                    decoded_p_observation_next = np.concatenate((np.full((F), 0), np.full((N*F), 0)))
                    for k in range(N*F):
                        decoded_p_observation_next[F+ k] = decoded_obs_next[N-I + k]
                        ii = k // F
                        f = k % F
                        H[ii][f] = decoded_obs_next[N-I + k]
                    for k in range(N-I):
                        f = decoded_obs_next[k]
                        C[i][f] = 1
                        decoded_p_observation_next[f] = 1
                    p_states_next[i] = encoder(decoded_p_observation_next, state_ranges_RSU)
                else: # Vehicules
                    decoded_p_observation_next = np.concatenate((np.full((1), 0), np.full((N*F), 0)))
                    f = decoded_obs_next[i-I]
                    decoded_p_observation_next[0] = f
                    C[i][f] = 1
                    for k in range(N*F):
                        decoded_p_observation_next[1+ k] = decoded_obs_next[N-I + k]
                        ii = k // F
                        f = k % F
                        H[ii][f] = decoded_obs_next[N-I + k]
                    p_states_next[i] = encoder(decoded_p_observation_next, state_ranges_vehicules)
            hits = np.sum(C*H) # if there is a request C=1, and available in cache H=1; it is a hit.
            misses = np.sum(C - H > 0) # if there is a request C=1, and unavailable in cache H=0; it is a miss
            flops = 0
            # Update Q-values with bellman equation
            for i in range(N):
                agent = agents[i]
                a = actions[i]
                r = reward
                eq_val = np.max(agent.q_table[p_states_next[i]]) # sellfish action decision, by maximizing its q value.
                agent.update_q(p_states[i], a, r, p_states_next[i], eq_val)
            observation = obs_next
            episode_reward += reward
            episode_flop += flops
            episode_hit += hits
            episode_miss += misses
            print("step duration {}".format(1000.0 * (time.time() - start_time)))
        print(f"[Python] Episode {episode} finished with reward: {episode_reward}")
        episode_rewards.append(episode_reward)
        episode_flops.append(episode_flop)
        episode_hits.append(episode_hit)
        episode_misses.append(episode_miss)
        with open(save_dir+ "metrics_"+ str(episode)+".pkl", "wb") as fd:
            pickle.dump((episode_rewards, episode_flops, episode_hits, episode_misses), fd)
        #if episode % 20 == 0: # saving numpay arras take big time, so we only save each 5 episodes (adjust if necessary)
        #    for i in range(N):
        #        agents[i].save(save_dir, episode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running MARL solution for edge caching jointly with veins simulation")
    parser.add_argument("--episode_start", type=int, required=True, help="From which to start episode")
    parser.add_argument("--nb_vehicules", type=int, required=True, help="Set number of vehicules in the simulation via --nb_vehicules")
    parser.add_argument("--nb_rsus", type=int, required=True, help="Set number of RSUs in simulation via --nb_rsus")
    parser.add_argument("--nb_content", type=int, required=True, help="Set the number of contents in simulation via --nb_content")
    args = parser.parse_args()
    episode_start = args.episode_start
    nb_vehicules = args.nb_vehicules
    nb_rsus = args.nb_rsus
    nb_content = args.nb_content
    main(episode_start=episode_start, nb_vehicules = nb_vehicules, nb_rsus = nb_rsus, nb_content = nb_content)

