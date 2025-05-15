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
from deap import base, creator, tools, algorithms
import random

CONFIG = {}

SUPPORT_ENUMERATION = 0
LEMKE_HOWSON = 1
FICTIOUS_PLAY = 2
QUANTAL_RESPONSE = 3

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

def compute_pareto_front_equilibrium(p_agents, obs, individual_actions_ranges):
    start_time = time.time()
    estimated_values = {agent.agent_id: 0.0 for agent in p_agents}
    counts = {agent.agent_id: 0 for agent in p_agents}
    individual_actions_sizes = np.zeros((len(individual_actions_ranges)))
    for i in range(len(individual_actions_sizes)):
        individual_actions_sizes[i] = np.prod(individual_actions_ranges[i])
    individual_actions_sizes = individual_actions_sizes.astype(np.int32)
    
    if len(p_agents) < 2:
        return np.max(p_agents[0].q_table[obs]), 0.0

    elif len(p_agents) == 2:  # simple case of two players
        print("Two players")
        a0, a1 = p_agents[0], p_agents[1]
        s0 = obs
        s1 = obs
        q1 = np.zeros((individual_actions_sizes[0], individual_actions_sizes[1]))    
        q2 = np.zeros((individual_actions_sizes[0], individual_actions_sizes[1]))
        for action in range(individual_actions_sizes[0] * individual_actions_sizes[1]):
            d_action = decoder(action, a0.action_ranges)
            d_action_p1 = [d_action[0], d_action[2]]
            d_action_p2 = [d_action[1], d_action[3]]
            action_p1 = encoder(d_action_p1, individual_actions_ranges[0])
            action_p2 = encoder(d_action_p2, individual_actions_ranges[1])
            q1[action_p1][action_p2] = a0.q_table[s0][action]
            q2[action_p1][action_p2] = a1.q_table[s1][action]
        # === NSGA-II setup ===
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))  # Maximize both rewards
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        toolbox = base.Toolbox()
        # Create individuals: [action_player1, action_player2]
        toolbox.register("attr_int_p1", random.randint, 0, individual_actions_sizes[0] - 1)
        toolbox.register("attr_int_p2", random.randint, 0, individual_actions_sizes[1] - 1)
        toolbox.register("individual", tools.initCycle, creator.Individual,
                         (toolbox.attr_int_p1, toolbox.attr_int_p2), n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Define fitness function
        def evaluate(individual):
            ap1, ap2 = individual
            payoff1 = q1[ap1][ap2]
            payoff2 = q2[ap1][ap2]
            return payoff1, payoff2

        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=[0, 0], up=[individual_actions_sizes[0]-1, individual_actions_sizes[1]-1], indpb=0.2)
        toolbox.register("select", tools.selNSGA2)

        pop = toolbox.population(n=50)

        # Evolution
        algorithms.eaMuPlusLambda(pop, toolbox, mu=50, lambda_=100, cxpb=0.7, mutpb=0.2,
                                  ngen=20, verbose=False)

        # Get Pareto front
        pareto_front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]

        # === Choose one point on the Pareto front (e.g., randomly, or maximize sum) ===
        best_ind = max(pareto_front, key=lambda ind: sum(ind.fitness.values))
        ap1, ap2 = best_ind
        vi = q1[ap1][ap2]
        vj = q2[ap1][ap2]

        estimated_values[a0.agent_id] += vi
        estimated_values[a1.agent_id] += vj
    else:
        num_agents = len(p_agents)
        print("{} players".format(num_agents))
        obs_list = [obs for _ in range(num_agents)]
        # Prepare payoff matrices
        q_is = []
        for idx in range(num_agents):
            qidx = np.zeros(individual_actions_sizes)
            q_is.append(qidx)
        
        for action in range(np.prod(individual_actions_sizes)):
            i_actions = {}
            d_action = decoder(action, p_agents[0].action_ranges)
            for i in range(len(p_agents)):
                d_i_actions = [d_action[i], d_action[i+len(p_agents)]]
                i_actions[i] = encoder(d_i_actions, individual_actions_ranges[i])
            for i in range(num_agents):
                q_is[i][tuple(i_actions)] = p_agents[i].q_table[obs_list[i]][action]
        
        creator.create("FitnessMulti", base.Fitness, weights=(1.0,) * num_agents)
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        toolbox = base.Toolbox()
        for idx in range(num_agents):
            toolbox.register(f"attr_int_p{idx}", random.randint, 0, individual_actions_sizes[idx] - 1)
        attrs = tuple(getattr(toolbox, f"attr_int_p{idx}") for idx in range(num_agents))
        toolbox.register("individual", tools.initCycle, creator.Individual, attrs, n=1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        def evaluate(individual): # each individual is (action1, action2, ..., actionN)
            joint_action = []
            for act in individual:
                joint_action.append(act)
            payoffs = []
            for idx in range(num_agents):
                payoffs.append(q_is[idx][tuple(joint_action)])
            return tuple(payoffs)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutUniformInt, low=[0] * num_agents, up=[s-1 for s in individual_actions_sizes], indpb=0.2)
        toolbox.register("select", tools.selNSGA2)
        
        pop = toolbox.population(n=100)
        algorithms.eaMuPlusLambda(pop, toolbox, mu=100, lambda_=200, cxpb=0.7, mutpb=0.2, ngen=30, verbose=False)
        pareto_front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
        # Choose best individual (maximize sum of rewards)
        best_ind = max(pareto_front, key=lambda ind: sum(ind.fitness.values))
        # Distribute rewards to estimated values
        rewards = best_ind.fitness.values
        for idx, agent in enumerate(p_agents):
            estimated_values[agent.agent_id] += rewards[idx]
    flops = 1000.0 * (time.time() - start_time)
    print("estimated values {}, flops {}".format(estimated_values, flops))
    return estimated_values, flops

def fictitious_play(A, B, iterations=1000):
    num_actions_p1, num_actions_p2 = A.shape
    # Initialize action counts (uniform initial belief)
    action_counts_p1 = np.ones(num_actions_p1)
    action_counts_p2 = np.ones(num_actions_p2)
    # Track mixed strategies over time (optional)
    history_p1 = []
    history_p2 = []
    for t in range(iterations):
        strategy_p2 = action_counts_p2 / np.sum(action_counts_p2)
        strategy_p1 = action_counts_p1 / np.sum(action_counts_p1)
        expected_util_p1 = A @ strategy_p2
        best_response_p1 = np.argmax(expected_util_p1)
        action_counts_p1[best_response_p1] += 1
        expected_util_p2 = B.T @ strategy_p1
        best_response_p2 = np.argmax(expected_util_p2)
        action_counts_p2[best_response_p2] += 1
        history_p1.append(strategy_p1.copy())
        history_p2.append(strategy_p2.copy())

    final_strategy_p1 = action_counts_p1 / np.sum(action_counts_p1)
    final_strategy_p2 = action_counts_p2 / np.sum(action_counts_p2)

    return final_strategy_p1, final_strategy_p2, history_p1, history_p2

def logit_response(payoffs, opponent_strategy, lam):
    """Compute logit response given payoffs and opponent strategy"""
    expected_utilities = payoffs @ opponent_strategy
    exp_utilities = np.exp(lam * expected_utilities)
    return exp_utilities / np.sum(exp_utilities)

def quantal_response_equilibrium(A, B, lam=1.0, iterations=1000, tol=1e-6):
    """Quantal Response Equilibrium for 2-player game"""
    n, m = A.shape
    sigma1 = np.ones(n) / n  # initial strategies
    sigma2 = np.ones(m) / m

    for _ in range(iterations):
        prev_sigma1 = sigma1.copy()
        prev_sigma2 = sigma2.copy()

        sigma1 = logit_response(A, sigma2, lam)
        sigma2 = logit_response(B.T, sigma1, lam)

        if np.allclose(sigma1, prev_sigma1, atol=tol) and np.allclose(sigma2, prev_sigma2, atol=tol):
            break

    return sigma1, sigma2


def compute_mf_nash_equilibrium(p_agents, obs, individual_actions_ranges, approach=SUPPORT_ENUMERATION):
    start_time = time.time()
    estimated_values = {agent.agent_id: 0.0 for agent in p_agents}
    counts = {agent.agent_id: 0 for agent in p_agents}
    individual_actions_sizes = np.zeros((len(individual_actions_ranges)))
    for i in range(len(individual_actions_sizes)):
        individual_actions_sizes[i] = np.prod(individual_actions_ranges[i])
    individual_actions_sizes = individual_actions_sizes.astype(np.int32)
    if len(p_agents) < 2:
        return np.max(p_agents[0].q_table[obs]), 0.0
    elif len(p_agents) == 2: # simple case of two players
        #print("Two players")
        a0, a1 = p_agents[0], p_agents[1]
        s0 = obs
        s1 = obs
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
        
        if approach == LEMKE_HOWSON:
            try:
                game = nash.Game(q1, q2)
                #print("Game players --LEMKE_HOWSON ({}, {})".format(q1.shape, q2.shape))
                equilibrium = game.lemke_howson(initial_dropped_label=0)
                pi1, pi2 = equilibrium
                vi = float(np.dot(pi1, np.dot(q1, pi2)))
                vj = float(np.dot(pi1, np.dot(q2, pi2)))
            except:
                #print("Game exception")
                vi = vj = 1.2
        
        elif approach == SUPPORT_ENUMERATION:
            try:
                game = nash.Game(q1, q2)
                #print("Game players --SUPPORT_ENUMERATION ({}, {})".format(q1.shape, q2.shape))
                support = game.support_enumeration()
                eq = list(support)[0]
                pi1, pi2 = eq
                vi = float(np.dot(pi1, np.dot(q1, pi2)))
                vj = float(np.dot(pi1, np.dot(q2, pi2)))
            except:
                #print("Game exception")
                vi = vj = 1.2
        
        elif approach == FICTIOUS_PLAY:
            #print("Game players --FICTIOUS_PLAY ({}, {})".format(q1.shape, q2.shape))
            pi1, pi2, h1, h2 = fictitious_play(q1, q2, iterations=1000)
            vi = float(np.dot(pi1, np.dot(q1, pi2)))
            vj = float(np.dot(pi1, np.dot(q2, pi2)))
            
        elif approach == QUANTAL_RESPONSE:
            #print("Game players --FICTIOUS_PLAY ({}, {})".format(q1.shape, q2.shape))
            pi1, pi2 = quantal_response_equilibrium(q1, q2, lam=1.0, iterations=1000, tol=1e-6)
            vi = float(np.dot(pi1, np.dot(q1, pi2)))
            vj = float(np.dot(pi1, np.dot(q2, pi2)))
        
            
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
            si = obs
            sj = obs
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
            
            if approach == LEMKE_HOWSON:
                #print("Game --LEMKE_HOWSON players (q1,q2) ({}, {})".format(qi.shape, qj.shape))
                try:
                    game = nash.Game(qi, qj)
                    equilibrium = game.lemke_howson(initial_dropped_label=0)
                    pi1, pi2 = equilibrium
                    #print("equilibrium {}".format(equilibrium))
                    vi = float(np.dot(pi1, np.dot(qi, pi2)))
                except:
                    #print("Game exception")
                    vi = 1.5
            
            elif approach == SUPPORT_ENUMERATION:
                #print("Game --SUPPORT_ENUMERATION players (q1,q2) ({}, {})".format(qi.shape, qj.shape))
                try:
                    game = nash.Game(qi, qj)
                    support = game.support_enumeration()
                    eq = list(support)[0]
                    pi1, pi2 = eq
                    #print("equilibrium {}".format(eq))
                    vi = float(np.dot(pi1, np.dot(qi, pi2)))
                except:
                    #print("Game exception")
                    vi = 1.5
            elif approach == FICTIOUS_PLAY:
                #print("Game --FICTIOUS_PLAY players (q1,q2) ({}, {})".format(qi.shape, qj.shape))
                #print("Game players --FICTIOUS_PLAY ({}, {})".format(q1.shape, q2.shape))
                pi1, pi2, h1, h2 = quantal_response_equilibrium(q1, q2, iterations=1000)
                vi = float(np.dot(pi1, np.dot(q1, pi2)))
                
            elif approach == QUANTAL_RESPONSE:
                #print("Game --QUANTAL_RESPONSE players (q1,q2) ({}, {})".format(qi.shape, qj.shape))
                #print("Game players --QUANTAL_RESPONSE ({}, {})".format(q1.shape, q2.shape))
                pi1, pi2 = fictitious_play(q1, q2, iterations=1000)
                vi = float(np.dot(pi1, np.dot(q1, pi2)))
            
            estimated_values[ai.agent_id] += vi
    flops = 1000.0 * (time.time() - start_time)
    #print("estimated values {}, flops {}".format(estimated_values, flops))
    return estimated_values, flops



def main(episode_start=0, nb_vehicules = 6, nb_rsus = 3, nb_content = 3):
    veins_gym_env = False
    save_dir = "./checkpoint/rsunash/"
    N = nb_rsus + nb_vehicules  # total agents (vehicles + RSUs)
    CONFIG["N"] = N
    I = nb_rsus  # number of RSUs (leaders)
    CONFIG["I"] = I
    F = nb_content  # content categories
    CONFIG["F"] = F
    H = np.zeros((I,F)) # N F, for each agent caching state varies from 0 to 1
    C = np.zeros((I,F)) # for each vehicule, values range from 0 to F-1; turning it into matrix shape make values range from 0 to 1 
    # all agents (RSU) actions
    action_ranges = np.concatenate((np.full((I), 2), np.full((I), F)))
    # all agents (RSU) state (queries, caching state)
    # caching state --> reduced to one vector which is the logical or between all rsus, this is accepted because 
    # rsus have wired connection between them, so if one rsu has at least content f, others will access to it easily
    # query state --> reduced to one vector, we only want to account for the rsu queries, others does not matter
    state_ranges = np.concatenate((np.full((F), 2), np.full((F), 2))) 
    #individual rsu action (for one rsu)
    individual_action_range = np.concatenate((np.full((1), 2), np.full((1), F)))
    
    all_action_ranges = np.concatenate((np.full((N - I), 2), np.full((N), 2), np.full((N), F)))
    all_state_ranges = np.concatenate((np.full((N - I), F), np.full((N*F), 2)))
    
    action_size = np.prod(action_ranges)
    
    state_size = np.prod(state_ranges)
    print("[Python] Agents state size {}, Agents action size {}".format(state_size, action_size))
    # the env accepts all (vehicules + rsu) state and actio sizes
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
        env = MockEnv(all_action_ranges, all_state_ranges)
    
    print("[Python] RSU MARL Training Start...")
    agents = {}
    START_FROM_EPISODE = episode_start
    
    for i in range(I):
        is_primary = True
        agents[i] = QLearningAgent(agent_id = i, state_ranges= state_ranges, action_ranges = action_ranges, is_primary = is_primary)
    
    episode_rewards = []
    episode_flops = []
    episode_hits = []
    episode_misses = []

    if START_FROM_EPISODE > 0:
        print("Loading ... {START_FROM_EPISODE}")
        for i in range(I):
            agents[i].load(save_dir, START_FROM_EPISODE)
        with open(save_dir + "metrics_"+str(START_FROM_EPISODE)+".pkl") as fd:
            episode_rewards, episode_flops, episode_hits, episode_misses = pickle.load(fd)    
    else:
        print("Training from scratch ... ")
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
            decoded_p_observation = np.concatenate((np.full((F), 0), np.full((F), 0)))
            for k in range(I*F):
                #decoded_p_observation[F+k] = decoded_obs[N-I + k]
                i = k // F
                f = k % F
                H[i][f] = decoded_obs[N-I + k]
            H_agg = np.sum(H, axis=0) >= 1
            for f in range(F):
                if H_agg[f]:
                    decoded_p_observation[F+f] = 1

            C = np.zeros((I,F))
            for k in range(N-I):
                f = decoded_obs[k]
                for i in range(I):
                    C[i][f] = 1
                decoded_p_observation[f] = 1
            p_observation = encoder(decoded_p_observation, state_ranges)
        
            # global action to send to environment
            action = np.concatenate((np.full((N - I), 0), np.full((N), 0), np.full((N), 0)))
            actions = {}
            for i in range(N):
                #now we will extract the actual agents action and put them in one action vector to send them to the environment.
                if(i < I): # primary agents only have caching decision and replacement decision
                    actions[i] = agents[i].choose_action(p_observation)
                    dec_action = decoder(actions[i], action_ranges)
                    action[N-I+i] = dec_action[i] # took caching decision.
                    action[2*N-I+i] = dec_action[I+i] # took the replacement decision.
                else: # secondary agents, will take target selection action, caching decision, caching replacement.
                    action[i] = 1 # default to V2R
                    action[N-I+i] = 0 # No cache in the vehicules
                    action[2*N-I+i] = 0 # this value is useless, just put a random value (0)
        
            # combine back into scalar
            encoded_action = encoder(action, all_action_ranges)
            # Step environment
            obs_next, reward, done, info = env.step(encoded_action)
            decoded_next_obs = decoder(obs_next, all_state_ranges)

            decoded_p_obs_next = np.concatenate((np.full((F), 0), np.full((F), 0)))
            for k in range(I*F):
                i = k // F
                f = k % F
                H[i][f] = decoded_next_obs[N-I + k]
            H_agg = np.sum(H, axis=0) >= 1
            for f in range(F):
                if H_agg[f]:
                    decoded_p_obs_next[F+f] = 1

            C = np.zeros((I,F))
            for k in range(N-I):
                f = decoded_next_obs[k]
                for i in range(I):
                    C[i][f] = 1
                decoded_p_obs_next[f] = 1
            
            hits = np.sum(C*H) # if there is a request C=1, and available in cache H=1; it is a hit.
            misses = np.sum(C - H > 0) # if there is a request C=1, and unavailable in cache H=0; it is a miss

            p_obs_next = encoder(decoded_p_obs_next, state_ranges)
            # Compute Nash Equilibrium for primary agents
            p_agents = [a for a in agents.values() if a.is_primary]
            individual_actions_ranges = [individual_action_range for a in agents.values() if a.is_primary]
            
            NASH = True
            if NASH:
                nash_values, flops = compute_mf_nash_equilibrium(p_agents, p_obs_next, individual_actions_ranges)
            else:
                nash_values, flops = compute_pareto_front_equilibrium(p_agents, p_obs_next, individual_actions_ranges)
            
            # Update Q-values
            for i in range(I):
                agent = agents[i]
                a = actions[i]
                r = reward
                eq_val = nash_values[i]
                agent.update_q(p_observation, a, r, p_obs_next, eq_val)
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
        #if episode % 10 == 0: # saving numpay arras take big time, so we only save each 5 episodes (adjust if necessary)
        #    for i in range(I):
        #       agents[i].save(save_dir, episode)

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
