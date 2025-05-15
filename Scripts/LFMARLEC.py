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

CONFIG = {}

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

class MockEnv(gym.Env):
    def __init__(self, action_ranges, state_ranges, episode_length=15):
        super(MockEnv, self).__init__()
        self.episode_length = episode_length
        self.step_count = 0
        self.tau = 10000 # in mS
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
        self.state = self.observation_space.sample()
        return self.state, reward, done, {}

class QLearningAgent:
    def __init__(self, agent_id, state_ranges, action_ranges, is_primary, alpha=0.1, gamma=0.95, epsilon=0.2):
        self.agent_id = agent_id
        self.action_ranges = action_ranges
        self.action_size = np.prod(action_ranges)
        self.is_primary = is_primary
        self.state_ranges = state_ranges
        self.state_size = np.prod(state_ranges)
        self.q_table = np.zeros((self.state_size, self.action_size))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    def load(self, save_dir, episode):
        print("Agent {} loading q table at episode {} ...".format(self.agent_id, episode))
        with open(save_dir+ "q_table_"+ str(self.agent_id)+ ".pkl", "rb") as fd:
            self.q_table = pickle.load(fd)
    def save(self, save_dir, episode):
        print("Agent {} saving q table at episode {} ...".format(self.agent_id, episode))
        with open(save_dir+ "q_table_"+ str(self.agent_id)+ ".pkl", "wb") as fd:
            pickle.dump(self.q_table, fd)
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return np.argmax(self.q_table[state])
    def update_q(self, state, action, reward, next_state, equilibrium_val):
        old_val = self.q_table[state][action]
        self.q_table[state][action] = (1 - self.alpha) * old_val + self.alpha * (reward + self.gamma * equilibrium_val)    
    @classmethod
    def average_agents(cls, agent_list, new_agent_id):
        if not agent_list:
            raise ValueError("Agent list is empty!")
        state_ranges = agent_list[0].state_ranges
        action_ranges = agent_list[0].action_ranges
        action_size = agent_list[0].action_size
        is_primary = agent_list[0].is_primary
        agent_id = new_agent_id
        new_agent = cls(agent_id, state_ranges, action_ranges, is_primary, alpha=0.1, gamma=0.95, epsilon=0.2)
        q_tables = []
        for agent in agent_list:
            q_tables.append(agent.q_table)
        q_tables = np.stack(q_tables)
        new_agent.q_table = np.mean(q_tables, axis=0)
        return new_agent

def compute_mf_nash_equilibrium(p_agents, obs, individual_actions_ranges):
    start_time = time.time()
    estimated_values = {agent: 0.0 for agent in p_agents}
    individual_actions_sizes = np.zeros((len(individual_actions_ranges)))
    for i in range(len(individual_actions_sizes)):
        individual_actions_sizes[i] = np.prod(individual_actions_ranges[i])
    individual_actions_sizes = individual_actions_sizes.astype(np.int32)
    if len(p_agents) < 2:
        return np.max(p_agents[0].q_table[obs]), 0.0
    elif len(p_agents) == 2: # simple case of two players
        #print("Two players")
        a0, a1 = p_agents[0], p_agents[1]
        s0 = obs[0]
        s1 = obs[1]
        q1 = np.zeros((individual_actions_sizes[0], individual_actions_sizes[1]))    
        q2 = np.zeros((individual_actions_sizes[0], individual_actions_sizes[1]))
        for action in range(individual_actions_sizes[0]*individual_actions_sizes[1]):
            d_action = decoder(action, a0.action_ranges)
            d_action_p1 = [d_action[0], d_action[2]]
            d_action_p2 = [d_action[1], d_action[3]]
            action_p1 = encoder(d_action_p1, individual_actions_ranges[0])
            action_p2 = encoder(d_action_p2, individual_actions_ranges[1])
            q1[action_p1][action_p2] = a0.q_table[s0][action]
            q2[action_p1][action_p2] = a1.q_table[s1][action]
        try:
            game = nash.Game(q1, q2)
            #print("Game players ({}, {})".format(q1.shape, q2.shape))
            support = game.support_enumeration()
            eq = list(support)[0]
            pi1, pi2 = eq
            vi = float(np.dot(pi1, np.dot(q1, pi2)))
            vj = float(np.dot(pi1, np.dot(q2, pi2)))
        except:
            print("Game exception")
            vi = vj = 1.2
        estimated_values[a0.agent_id] += vi
        estimated_values[a1.agent_id] += vj
    else:
        #print("{} Players".format(len(p_agents)))
        other_action_ranges = np.concatenate((np.full((CONFIG["I"]-1), 2), np.full((CONFIG["I"]-1), CONFIG["F"])))    
        for i in range(len(p_agents)):
            ai = p_agents[i]
            other_agents = []
            for j in range(len(p_agents)):
                if j!=i:
                    other_agents.append(p_agents[j])
            aj = QLearningAgent.average_agents(other_agents, new_agent_id=100)
            si = obs[i]
            sj = obs[i]
            qi = np.zeros((individual_actions_sizes[0], individual_actions_sizes[0]**(CONFIG["I"]-1))) # it is asymmetric game between rsus and MF agent.
            qj = np.zeros((individual_actions_sizes[0], individual_actions_sizes[0]**(CONFIG["I"]-1))) # MF agent
            for action in range(individual_actions_sizes[0]**len(p_agents)):
                d_action = decoder(action, ai.action_ranges)
                d_action_p1 = [d_action[i], d_action[i+CONFIG["I"]]]
                d_action_p2 = np.concatenate((np.full((CONFIG["I"]-1), 0), np.full((CONFIG["I"]-1), 0)))
                d_action_p2 = np.delete(np.asarray(d_action), [i,i+CONFIG["I"]])
                action_p1 = encoder(d_action_p1, individual_actions_ranges[i])
                action_p2 = encoder(d_action_p2, other_action_ranges)
                qi[action_p1][action_p2] = ai.q_table[si][action]
                qj[action_p1][action_p2] = aj.q_table[sj][action]
            #print("Game 1 players (q1,q2) ({}, {})".format(qi.shape, qj.shape))
            try:
                game = nash.Game(qi, qj)
                equilibrium = game.lemke_howson(initial_dropped_label=0)
                pi1, pi2 = equilibrium
                #print("equilibrium {}".format(equilibrium))
                vi = float(np.dot(pi1, np.dot(qi, pi2)))
            except:
                print("Game exception")
                vi = 1.5
            estimated_values[ai.agent_id] += vi
    flops = 1000.0 * (time.time() - start_time)
    #print("estimated values {}, flops {}".format(estimated_values, flops))
    return estimated_values, flops

def compute_stackelberg_equilibrium(p_agents, s_agents, p_individual_actions_ranges, s_individual_actions_ranges, p_obs, s_obs, neighborhouds): # neighborhouds are dictionary for each index of vehicules, it gives index of rsu. if no rsu is in range of vehicule, vehicule take sellfish action
    start_time = time.time()
    estimated_values = {agent: 0.0 for agent in s_agents} # we only need s agents estimated values
    
    p_individual_actions_sizes = np.zeros((len(p_individual_actions_ranges)))
    s_individual_actions_sizes = np.zeros((len(s_individual_actions_ranges)))
    for i in range(len(p_individual_actions_sizes)):
        p_individual_actions_sizes[i] = np.prod(p_individual_actions_ranges[i])
    p_individual_actions_sizes = p_individual_actions_sizes.astype(np.int32)
    p_individual_actions_sizes = {p_agent: p_individual_actions_sizes[p_agent] for p_agent in p_agents}
    
    II = len(p_agents)
    for i in range(len(s_individual_actions_sizes)):
        s_individual_actions_sizes[i] = np.prod(s_individual_actions_ranges[i+II])
    s_individual_actions_sizes = s_individual_actions_sizes.astype(np.int32)
    s_individual_actions_sizes = {s_agent: s_individual_actions_sizes[s_agent-II] for s_agent in s_agents}
    
    if len(s_agents) < 1:
        return {}, 0.0
    for s_agent in s_agents.values():
        idx_p_agent = neighborhouds[s_agent.agent_id]
        p_agent = p_agents[idx_p_agent]
        ss = s_obs[s_agent.agent_id]
        sp = p_obs[p_agent.agent_id]
        qs = np.zeros((s_individual_actions_sizes[s_agent.agent_id], p_individual_actions_sizes[p_agent.agent_id]))    
        qp = np.zeros((s_individual_actions_sizes[s_agent.agent_id], p_individual_actions_sizes[p_agent.agent_id]))
        
        for s_action in range(s_agent.action_size):
            d_s_action = decoder(s_action, s_agent.action_ranges)
            i_d_s_action = [d_s_action[0], d_s_action[1], d_s_action[II+2]]
            i_s_action = encoder(i_d_s_action, s_individual_actions_ranges[s_agent.agent_id])
            i_d_p_action = [d_s_action[2+p_agent.agent_id], d_s_action[3+II+p_agent.agent_id]]
            i_p_action = encoder(i_d_p_action, p_individual_actions_ranges[p_agent.agent_id])
            qs[i_s_action][i_p_action] += s_agent.q_table[ss][s_action]
        for p_action in range(p_agent.action_size):
            d_p_action = decoder(p_action, p_agent.action_ranges)
            i_d_p_action = [d_p_action[p_agent.agent_id], d_p_action[II+p_agent.agent_id]]
            i_p_action = encoder(i_d_p_action, p_individual_actions_ranges[p_agent.agent_id])
            for i_s_action in range(s_individual_actions_sizes[s_agent.agent_id]):
                qp[i_s_action][i_p_action] += p_agent.q_table[sp][p_action]
        
        # after preparing the game matrices qs and qp, calculate the stackelberg equilibrium (backward induction)
        best_leader_payoff = -np.inf
        best_leader_action = None
        best_follower_action = None
        num_follower_actions, num_leader_actions = qs.shape
        
        for leader_action in range(num_leader_actions):
            # Follower best response for this leader action
            follower_payoffs = qs[:, leader_action]
            follower_best_action = np.argmax(follower_payoffs)
            # Leader payoff for this pair
            leader_payoff = qp[follower_best_action, leader_action]
            if leader_payoff > best_leader_payoff:
                best_leader_payoff = leader_payoff
                best_leader_action = leader_action
                best_follower_action = follower_best_action
        estimated_values[s_agent.agent_id] = qs[best_follower_action, best_leader_action]
        
    flops = 1000.0 * (time.time() - start_time)
    print("stackelberg estimated values {}, flops {}".format(estimated_values, flops))
    return estimated_values, flops

def main(episode_start=0, nb_vehicules = 6, nb_rsus = 3, nb_content = 3):
    veins_gym_env = False
    save_dir = "./checkpoint/stackelberg/"
    N = nb_rsus + nb_vehicules  # total agents (vehicles + RSUs)
    CONFIG["N"] = N
    I = nb_rsus  # number of RSUs (leaders)
    CONFIG["I"] = I
    F = nb_content  # content categories
    CONFIG["F"] = F
    H = np.zeros((N,F)) # N F, for each agent caching state varies from 0 to 1
    C = np.zeros((N,F)) # for each vehicule, values range from 0 to F-1; turning it into matrix shape make values range from 0 to 1 
    ## Leaders (RSU) actions
    action_ranges_RSU = np.concatenate((np.full((I), 2), np.full((I), F)))
    ## Leaders (RSU) state (queries, caching state)
    state_ranges_RSU = np.concatenate((np.full((F), 2), np.full((F), 2))) # first part own requests; second part: logical or
    individual_action_range_RSU = np.concatenate((np.full((1), 2), np.full((1), F)))
    action_size_RSU = np.prod(action_ranges_RSU)
    state_size_RSU = np.prod(state_ranges_RSU)

    ## Followers (Vehicules) state
    state_ranges_vehicule = np.concatenate((np.full((1), F), np.full((F), 2), np.full((F), 2))) # vehicule has only observation of its request thus (1,F); I+1 means in addition to all rsu caching state, it has also its own requests and caching state
    ## Followers (Vehicules) action
    action_ranges_vehicule = np.concatenate((np.full((1), 2), np.full((I+1), 2), np.full((I+1), F))) # v2v; cache or not; replacement 
    action_size_vehicules = np.prod(action_ranges_vehicule)
    state_size_vehicules = np.prod(state_ranges_vehicule)
    individual_action_range_vehicule = np.concatenate((np.full((1), 2), np.full((1), 2), np.full((1), F)))

    all_action_ranges = np.concatenate((np.full((N - I), 2), np.full((N), 2), np.full((N), F)))
    all_state_ranges = np.concatenate((np.full((N - I), F), np.full((N*F), 2)))
    print("[Python] Vehicule state size {}, action size {}".format(state_size_vehicules, action_size_vehicules))
    print("[Python] RSU state size {}, action size {}".format(state_size_RSU, action_size_RSU))
    
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
        env = MockEnv(all_action_ranges, all_state_ranges) # this is just for initial testing
    print("[Python] LFMARLEC Training Start...")
    agents = {}
    p_agents = {}
    s_agents = {}
    START_FROM_EPISODE = episode_start
    for i in range(N):
        if i < I:
            agents[i] = QLearningAgent(agent_id = i, state_ranges= state_ranges_RSU, action_ranges = action_ranges_RSU, is_primary = True)
            p_agents[i] = agents[i]
        else:
            agents[i] = QLearningAgent(agent_id = i, state_ranges= state_ranges_vehicule, action_ranges = action_ranges_vehicule, is_primary = False)
            s_agents[i] = agents[i]

    episode_rewards = []
    episode_flops = []
    episode_hits = []
    episode_misses = []

    if START_FROM_EPISODE > 0:
        print("Loading ... {START_FROM_EPISODE}")
        for i in range(N):
            agents[i].load(save_dir, START_FROM_EPISODE)
            if(i < I):
                p_agents[i] = agents[i]
            else:
                s_agents[i] = agents[i]
        with open(save_dir + "metrics_"+str(START_FROM_EPISODE)+".pkl") as fd:
            episode_rewards, episode_flops, episode_hits, episode_misses = pickle.load(fd)    
    else:
        print("Training Stackelberg from scratch ...")
    
    for episode in range(START_FROM_EPISODE, 50):
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
            # we construct the partial observations for the agents (RSUs)
            p_decoded_p_observation = np.concatenate((np.full((F), 0), np.full((F), 0)))
            s_decoded_p_observations = {ss: np.concatenate((np.full((1), 0), np.full((F), 0), np.full((F), 0))) for ss in s_agents}
            for k in range(N*F): #
                i = k // F
                f = k % F
                H[i][f] = decoded_obs[N-I + k]
            
            caches_rsu = np.any(H[:I, :], axis=0).astype(int)
            
            for k in range(N-I):
                f = decoded_obs[k]
                C[k+I][f] = 1
                s_decoded_p_observations[k+I][0] = f
            requests_rsu = np.any(C[I:, :], axis=0).astype(int)
            for i in range(I):
                for f in range(F):
                    C[i][f] = requests_rsu[f]
            
            for f in range(F):
                p_decoded_p_observation[F+f] = caches_rsu[f]
                p_decoded_p_observation[f] = requests_rsu[f]
                for s in s_agents:
                    s_decoded_p_observations[s][1+F+f] = caches_rsu[f]
                    s_decoded_p_observations[s][1+f] = H[s][f]
            p_p_observations = {p: encoder(p_decoded_p_observation, state_ranges_RSU) for p in p_agents}
            s_p_observations = {s: encoder(s_decoded_p_observations[s], state_ranges_vehicule) for s in s_agents}
                            
            # global action to send to environment
            action = np.concatenate((np.full((N - I), 0), np.full((N), 0), np.full((N), 0)))
            actions = {}
            for i in range(N):
                #now we will extract the actual agents action and put them in one action vector to send them to the environment.
                if(i < I): # primary agents only have caching decision and replacement decision
                    actions[i] = agents[i].choose_action(p_p_observations[i])
                    dec_action = decoder(actions[i], action_ranges_RSU)
                    action[N-I+i] = dec_action[i] # took caching decision.
                    action[2*N-I+i] = dec_action[I+i] # took the replacement decision.
                else: # secondary agents, will take target selection action, caching decision, caching replacement.
                    actions[i] = agents[i].choose_action(s_p_observations[i])
                    dec_action = decoder(actions[i], action_ranges_vehicule)
                    action[i-I] = dec_action[0]
                    action[N-I+i] = dec_action[1]
                    action[2*N-I+i] = dec_action[I+2]
        
            # combine back into scalar
            encoded_action = encoder(action, all_action_ranges)
            # Step environment
            obs_next, reward, done, info = env.step(encoded_action)
            
            H = np.zeros((N,F))
            C = np.zeros((N,F))
            
            decoded_obs_next = decoder(obs_next, all_state_ranges)
            # we construct the partial observations for the agents (RSUs)
            p_decoded_p_observation_next = np.concatenate((np.full((F), 0), np.full((F), 0)))
            s_decoded_p_observations_next = {s: np.concatenate((np.full((1), 0), np.full((F), 0), np.full((F), 0))) for s in s_agents}
            for k in range(N*F): #
                i = k // F
                f = k % F
                H[i][f] = decoded_obs_next[N-I + k]
            
            caches_rsu = np.any(H[:I, :], axis=0).astype(int)
            
            for k in range(N-I):
                f = decoded_obs_next[k]
                C[k+I][f] = 1
                s_decoded_p_observations_next[k+I][0] = f
            requests_rsu = np.any(C[I:, :], axis=0).astype(int)
            for i in range(I):
                for f in range(F):
                    C[i][f] = requests_rsu[f]
            
            for f in range(F):
                p_decoded_p_observation_next[F+f] = caches_rsu[f]
                p_decoded_p_observation_next[f] = requests_rsu[f]
                for s in s_agents:
                    s_decoded_p_observations_next[s][1+F+f] = caches_rsu[f]
                    s_decoded_p_observations_next[s][1+f] = H[s][f]
            p_p_observations_next = {p: encoder(p_decoded_p_observation_next, state_ranges_RSU) for p in p_agents}
            s_p_observations_next = {s: encoder(s_decoded_p_observations_next[s], state_ranges_vehicule) for s in s_agents}
            
            hits = np.sum(C*H) # if there is a request C=1, and available in cache H=1; it is a hit.
            misses = np.sum(C - H > 0) # if there is a request C=1, and unavailable in cache H=0; it is a miss

            p_individual_actions_ranges = {p: individual_action_range_RSU for p in p_agents}
            s_individual_actions_ranges = {s: individual_action_range_vehicule for s in s_agents}
            neighborhouds = {s: 0 for s in s_agents}
            # Compute Nash and Stackelberg Equilibria for the corresponding agents
            nash_values, flops1 = compute_mf_nash_equilibrium(p_agents, p_p_observations_next, p_individual_actions_ranges)
            stackelberg_values, flops2 = compute_stackelberg_equilibrium(p_agents, s_agents, p_individual_actions_ranges, s_individual_actions_ranges, p_p_observations_next, s_p_observations_next, neighborhouds)
            flops = flops1 + flops2
            # Update Q-values
            for p_agent in p_agents:
                p_agent = p_agents[p_agent]
                a = actions[p_agent.agent_id]
                r = reward
                eq_val = nash_values[p_agent.agent_id]
                p_agent.update_q(p_p_observations[p_agent.agent_id], a, r, p_p_observations_next[p_agent.agent_id], eq_val)
            
            for s_agent in s_agents:
                s_agent = s_agents[s_agent]
                a = actions[s_agent.agent_id]
                r = reward
                eq_val = stackelberg_values[s_agent.agent_id]
                s_agent.update_q(s_p_observations[s_agent.agent_id], a, r, s_p_observations_next[s_agent.agent_id], eq_val)
              
            observation = obs_next
            episode_reward += reward
            episode_flop += flops
            episode_hit += hits
            episode_miss += misses
            print("step duration {}".format(1000.0 * (time.time() - start_time)))
        #print(f"[Python] Episode {episode} finished with reward: {episode_reward}")
        episode_rewards.append(episode_reward)
        episode_flops.append(episode_flop)
        episode_hits.append(episode_hit)
        episode_misses.append(episode_miss)
        with open(save_dir+ "metrics_"+ str(episode)+".pkl", "wb") as fd:
            pickle.dump((episode_rewards, episode_flops, episode_hits, episode_misses), fd)
        if episode % 20 == 0: # saving numpay arras take big time, so we only save each 5 episodes (adjust if necessary)
            for i in range(N):
                agents[i].save(save_dir, episode)

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
