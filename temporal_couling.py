import numpy as np, time
#import MultiNEAT as NEAT
import mod_temporal_coupling as mod, sys
from random import randint#, choice
#from copy import deepcopy



viz = 0
gen_traj = 0
if viz:
    mod.vizualize_trajectory()
    sys.exit()

class Parameters:
    def __init__(self):

        self.aamas_domain = 1
        if self.aamas_domain == 1:
            #Plastic vars
            self.D_reward = 1
            self.num_agent_failure = 3

            #Constant Vars
            if True:
                self.sensor_noise = 0
                self.is_hard_time_offset = 1
                self.is_fuel = 0
                self.is_poi_move = 1  # POI move periodically
                self.wheel_action = 0
                self.poi_motion_probability = 0.25  # [0.0-1.0] probability
                self.time_offset = 3
                self.population_size = 100
                self.grid_row = 20
                self.grid_col = 20
                self.total_steps = 25  # Total roaming steps without goal before termination
                self.num_agents_scout = 4
                self.num_agents_service_bot = 8
                self.num_poi = 12
                self.agent_random = 0
                self.poi_random = 1
                self.is_time_offset = 1
                self.domain_setup = 0
                self.periodic_poi = 1  # Implements a 2 step POI movement

                # If no time offset the activation time is set to total length of episode
                if self.is_time_offset == 0:  # DO NOT CHANGE THE IF LOOP, alter only ELSE part
                    self.time_offset = self.total_steps;
                    self.is_hard_time_offset = 0


        else: #NOT AAMAS
            self.population_size = 100
            self.D_reward = 1  # D reward scheme
            self.grid_row = 20
            self.grid_col = 20
            self.total_steps = 25 # Total roaming steps without goal before termination
            self.num_agents_scout = 4
            self.num_agents_service_bot = 8
            self.num_poi = 12
            self.agent_random = 0
            self.poi_random = 1

            self.wheel_action = 1

            self.is_time_offset = 1 #Controls whether there is a time-offset or not

            #If no time offset the activation time is set to total length of episode
            if self.is_time_offset == 0: #DO NOT CHANGE THE IF LOOP, alter only ELSE part
                self.time_offset = self.total_steps; self.is_hard_time_offset = 0
            else: #CHANGE THIS

                self.time_offset = 2 #The time-offset / window for observation to happen following scouting
                self.is_hard_time_offset = 1 #Controls whether the time offset is hard or conversely soft

            self.is_fuel = 0  # Binary deciding whether to have fuel cost as a consideration


            #POI motion
            self.is_poi_move = 1  # POI move randomly
            self.periodic_poi = 1 #Implements a 2 step POI movement
            self.poi_motion_probability = 0.25 #[0.0-1.0] probability


            #Predefined domains for testing
            self.domain_setup = 0
            if self.domain_setup != 0:
                if self.domain_setup == 2:
                    self.num_poi = 4
                    self.num_agents_scout = 2
                    self.num_agents_service_bot = 4
                    self.total_steps = 3
                    self.is_time_offset = 1
                    self.is_hard_time_offset = 1
                    self.time_offset = 2
                    self.poi_random = 0
                    self.agent_random = 0
                    self.grid_col = 14; self.grid_row = 14
                    self.is_poi_move = 0

                elif self.domain_setup == 3:
                    self.num_poi = 2
                    self.num_agents_scout = 1
                    self.num_agents_service_bot = 2
                    self.total_steps = 3
                    self.is_time_offset = 1
                    self.is_hard_time_offset = 0
                    self.time_offset = 3
                    self.poi_random = 0
                    self.agent_random = 0
                    self.grid_col = 9; self.grid_row = 9
                    self.is_poi_move = 0



        #Tertiary Variables (Closed for cleanliness) Open to modify
        if True:
            # GIVEN
            self.is_service_cost = 1  # Cost for service bots to service a POI
            self.total_generations = 1000
            self.reward_scheme = 3  # 1: Order not relevant and is_scouted not required
            # 2: Order not relevant and is_scouted required
            # 3: Order relevant and is_scouted required

            #TERTIARY
            self.angle_res = 45
            self.use_neat = 1  # Use NEAT VS. Keras based Evolution module
            self.obs_dist = 1  # Observe distance (Radius of POI that agents have to be in for successful observation)
            self.coupling = 1  # Number of agents required to simultaneously observe a POI
            self.use_py_neat = 0  # Use Python implementation of NEAT
            self.predictor_hnodes = 20  # Prediction module hidden nodes (also the input to the Evo-net)
            self.augmented_input = 1
            self.baldwin = 0
            self.online_learning = 1
            self.update_sim = 1
            self.pre_train = 0

            self.sim_all = 0  # Simulator learns to predict the enmtrie state including the POI's
            self.share_sim_subpop = 1  # Share simulator within a sub-population
            self.sensor_avg = True  # Average distance as state input vs (min distance by default)
            self.split_learner = 0
            self.state_representation = 1  # 1 --> Angled brackets, 2--> List of agent/POIs
            if self.split_learner: self.state_representation = 1 #ENSURE
            self.use_rnn = 0  # Use recurrent instead of normal network
            #self.success_replay = False
            #self.vizualize = False

            # Determine Evo-input size
            self.evo_input_size = 0
            if self.state_representation == 2:
                if self.baldwin and self.augmented_input:
                    self.evo_input_size = self.predictor_hnodes + self.num_agents * 2 + self.num_poi * 2
                elif self.baldwin and not self.augmented_input:
                    self.evo_input_size = self.predictor_hnodes
                else:
                    self.evo_input_size = self.num_agents * 2 + self.num_poi * 2
            elif self.state_representation == 1:
                if self.baldwin and self.augmented_input:
                    self.evo_input_size = self.predictor_hnodes + (360 * 4 / self.angle_res)
                elif self.baldwin and not self.augmented_input:
                    self.evo_input_size = self.predictor_hnodes
                else:
                    self.evo_input_size = (360 * 6 / self.angle_res)
            elif self.state_representation == 3:
                k = 1
                #TODO Complete this
            if self.split_learner and not self.baldwin: #Strict Darwin with no Baldwin for split learner
                self.evo_input_size = self.num_agents * 2 + self.num_poi * 2 + (360 * 4 / self.angle_res)
            #print self.evo_input_size

            #EV0-NET
            self.use_hall_of_fame = 0
            self.hof_weight = 0.99
            self.leniency = 1  # Fitness calculation based on leniency vs averaging

            if self.use_neat: #Neat
                if self.use_py_neat: #Python NEAT
                    self.py_neat_config = Py_neat_params()

                else: #Multi-NEAT parameters
                    import MultiNEAT as NEAT
                    self.params = NEAT.Parameters()
                    self.params.PopulationSize = self.population_size
                    self.params.fs_neat = 1
                    self.params.evo_hidden = 10
                    self.params.MinSpecies = 5
                    self.params.MaxSpecies = 15
                    self.params.EliteFraction = 0.05
                    self.params.RecurrentProb = 0.2
                    self.params.RecurrentLoopProb = 0.2

                    self.params.MaxWeight = 8
                    self.params.MutateAddNeuronProb = 0.01
                    self.params.MutateAddLinkProb = 0.05
                    self.params.MutateRemLinkProb = 0.01
                    self.params.MutateRemSimpleNeuronProb = 0.005
                    self.params.MutateNeuronActivationTypeProb = 0.005

                    self.params.ActivationFunction_SignedSigmoid_Prob = 0.01
                    self.params.ActivationFunction_UnsignedSigmoid_Prob = 0.5
                    self.params.ActivationFunction_Tanh_Prob = 0.1
                    self.params.ActivationFunction_SignedStep_Prob = 0.1
            else: #Use keras
                self.keras_evonet_hnodes = 25  # Keras based Evo-net's hidden nodes

class Py_neat_params:
    def __init__(self):

        #[Types]
        self.stagnation_type = 'DefaultStagnation'
        self.reproduction_type = 'DefaultReproduction'

        #Phenotype
        self.input_nodes = 21
        self.hidden_nodes = 0
        self.output_nodes = 19
        self.initial_connection = 'fs_neat' #['unconnected', 'fs_neat', 'fully_connected', 'partial']
        self.max_weight = 0.1
        self.min_weight = -0.1
        self.feedforward = 0
        self.activation_functions = 'sigmoid'
        self.weight_stdev = 0.1

        #genetic
        self.pop_size = 500
        self.max_fitness_threshold = 1000000
        self.prob_add_conn = 0.1
        self.prob_add_node = 0.05
        self.prob_delete_conn = 0.01
        self.prob_delete_node = 0.01
        self.prob_mutate_bias = 0.05
        self.bias_mutation_power = 1.093
        self.prob_mutate_response = 0.1
        self.response_mutation_power = 0.1
        self.prob_mutate_weight = 0.2
        self.prob_replace_weight = 0.05
        self.weight_mutation_power = 1
        self.prob_mutate_activation = 0.08
        self.prob_toggle_link = 0.05
        self.reset_on_extinction = 1

        #genotype compatibility
        self.compatibility_threshold = 3.0
        self.excess_coefficient = 1.4
        self.disjoint_coefficient = 1.3
        self.weight_coefficient = 0.7

        self.species_fitness_func = 'mean'
        self.max_stagnation = 15

        self.elitism = 1
        self.survival_threshold = 0.2

parameters = Parameters() #Create the Parameters class
tracker = mod.statistics(parameters) #Initiate tracker
gridworld = mod.Gridworld (parameters)  # Create gridworld

if parameters.use_hall_of_fame: hof_util = mod.Hof_util()

def test_policies(save_name = 'trajectory.csv'):
    gridworld.load_test_policies() #Load test policies from best_team folder Assumes perfect sync
    fake_team = np.zeros(parameters.num_agents_scout + parameters.num_agents_service_bot)  # Fake team for test_phase
    best_reward = -10000
    for i in range(parameters.population_size):
        reward, global_reward, trajectory_log  = run_simulation(parameters, gridworld, fake_team, is_test=True)
        if global_reward > best_reward:
            comment = [parameters.num_agents_scout, parameters.num_agents_service_bot, parameters.num_poi] + [0] * (len(trajectory_log[0])-3)
            trajectory_log = [comment] + trajectory_log
            trajectory_log = np.array(trajectory_log)
            np.savetxt(save_name, trajectory_log, delimiter=',',fmt='%10.5f')
            print reward, global_reward
            best_reward = global_reward

def best_performance_trajectory(parameters, gridworld, teams, save_name='best_performance_traj.csv'):
    trajectory_log = []
    gridworld.reset(teams)  # Reset board
    # mod.dispGrid(gridworld)
    for steps in range(parameters.total_steps):  # One training episode till goal is not reached
        ig_traj_log = []
        for id, agent in enumerate(gridworld.agent_list_scout):  # get all the action choices from the agents
            if steps == 0: agent.perceived_state = gridworld.get_state(agent, is_Scout=True)  # Update all agent's perceived state
            agent.take_action(teams[id])  # Make the agent take action using the Evo-net with given id from the population
            ig_traj_log.append(agent.position[0])
            ig_traj_log.append(agent.position[1])

        for id, agent in enumerate(gridworld.agent_list_service_bot):  # get all the action choices from the agents
            if steps == 0: agent.perceived_state = gridworld.get_state(agent, is_Scout=False)  # Update all agent's perceived state
            agent.take_action(teams[id + parameters.num_agents_scout])  # Make the agent take action using the Evo-net with given id from the population
            ig_traj_log.append(agent.position[0])
            ig_traj_log.append(agent.position[1])



        for poi in gridworld.poi_list:
            ig_traj_log.append(poi.position[0])
            ig_traj_log.append(poi.position[1])

        trajectory_log.append(np.array(ig_traj_log))
        gridworld.move()  # Move gridworld

        # mod.dispGrid(gridworld)
        # raw_input('E')

        gridworld.update_poi_observations()  # Figure out the POI observations and store all credit information
        if parameters.is_poi_move: gridworld.poi_move()

        for id, agent in enumerate(gridworld.agent_list_scout): agent.referesh(teams[id], gridworld)  # Update state and learn if applicable
        for id, agent in enumerate(gridworld.agent_list_service_bot): agent.referesh(teams[id], gridworld)  # Update state and learn if applicable

        if gridworld.check_goal_complete(): break  # If all POI's observed

    # Log final position
    ig_traj_log = []
    for agent in gridworld.agent_list_scout:
        ig_traj_log.append(agent.position[0])
        ig_traj_log.append(agent.position[1])

    for agent in gridworld.agent_list_service_bot:
        ig_traj_log.append(agent.position[0])
        ig_traj_log.append(agent.position[1])

    for poi in gridworld.poi_list:
        ig_traj_log.append(poi.position[0])
        ig_traj_log.append(poi.position[1])
    trajectory_log.append(np.array(ig_traj_log))
    rewards, global_reward = gridworld.get_reward(teams)

    #Save trajectory to file
    comment = [parameters.num_agents_scout, parameters.num_agents_service_bot, parameters.num_poi] + [0] * (
    len(trajectory_log[0]) - 3)
    trajectory_log = [comment] + trajectory_log
    trajectory_log = np.array(trajectory_log)
    np.savetxt(save_name, trajectory_log, delimiter=',', fmt='%10.5f')


num_evals = 5
def evolve(gridworld, parameters, generation, best_hof_score):
    best_team = None; best_global = -100; epoch_best_team = None
    #Reset initial random positions for the epoch
    gridworld.new_epoch_reset()

    #NOTE: ALL keyword means all sub-populations

    # Get new genome list and fitness evaluations trackers
    for i in range(parameters.num_agents_scout): gridworld.agent_list_scout[i].evo_net.referesh_genome_list()
    for i in range(parameters.num_agents_service_bot): gridworld.agent_list_service_bot[i].evo_net.referesh_genome_list()

    #MAKE SELECTION POOLS
    teams = np.zeros(parameters.num_agents_scout + parameters.num_agents_service_bot).astype(int) #Team definitions by index
    selection_pool = []; max_pool_size = 0#Selection pool listing the individuals with multiples for to match number of evaluations
    for i in range(parameters.num_agents_scout): #Filling the selection pool
        if parameters.use_neat: ig_num_individuals = len(gridworld.agent_list_scout[i].evo_net.genome_list) #NEAT's number of individuals can change
        else: ig_num_individuals = parameters.population_size #For keras_evo-net the number of individuals stays constant at population size
        selection_pool.append(np.arange(ig_num_individuals))
        for j in range(num_evals - 1): selection_pool[i] = np.append(selection_pool[i], np.arange(ig_num_individuals))
        if len(selection_pool[i]) > max_pool_size: max_pool_size = len(selection_pool[i])
    for i in range(parameters.num_agents_service_bot): #Filling the selection pool
        if parameters.use_neat: ig_num_individuals = len(gridworld.agent_list_service_bot[i].evo_net.genome_list) #NEAT's number of individuals can change
        else: ig_num_individuals = parameters.population_size #For keras_evo-net the number of individuals stays constant at population size
        selection_pool.append(np.arange(ig_num_individuals))
        for j in range(num_evals - 1): selection_pool[i+parameters.num_agents_scout] = np.append(selection_pool[i+parameters.num_agents_scout], np.arange(ig_num_individuals))
        if len(selection_pool[i+parameters.num_agents_scout]) > max_pool_size: max_pool_size = len(selection_pool[i+parameters.num_agents_scout])

    for i, pool in enumerate(selection_pool): #Equalize the selection pool
        diff = max_pool_size - len(pool)
        if diff != 0:
            ig_cap = len(pool) / num_evals
            while diff > ig_cap:
                selection_pool[i] = np.append(selection_pool[i], np.arange(ig_cap))
                diff -= ig_cap
            selection_pool[i] = np.append(selection_pool[i], np.arange(diff))


    #MAIN LOOP
    for evals in range(parameters.population_size * num_evals): #For all evaluation cycles
        #PICK TEAMS
        for i in range(len(teams)):
            debug_choice_bound = (len(selection_pool[i])-1)
            if debug_choice_bound == 0:
                rand_index = 0
            else:
                rand_index = randint(0, debug_choice_bound) #Get a random index
            teams[i] = selection_pool[i][rand_index] #Pick team member from that index
            selection_pool[i] = np.delete(selection_pool[i], rand_index) #Delete that index

        #BUILD NETS
        for i in range(len(teams)):
            if i < parameters.num_agents_scout: #Scouts
                gridworld.agent_list_scout[i].evo_net.build_net(teams[i])  # build network for the genomes within the team
            else: #service_bot
                gridworld.agent_list_service_bot[i-parameters.num_agents_scout].evo_net.build_net(teams[i])  # build network for the genomes within the team

        #ION AND TRACK REWARD
        rewards, global_reward = run_simulation(parameters, gridworld, teams) #Returns rewards for each member of the team
        if global_reward > best_global:
            epoch_best_team = np.copy(teams)  # Store the best team from current epoch
            best_global = global_reward; #Store the best global performance
            if global_reward > best_hof_score:
                best_team = np.copy(teams) #Store the best team ever seen (HOF)
                best_performance_trajectory(parameters, gridworld, teams)


        #ENCODE FITNESS BACK TO AGENT and ALSO consider fuel cost
        for id, agent in enumerate(gridworld.agent_list_scout):
            ig_reward = rewards[id]
            #if parameters.is_fuel: ig_reward = rewards[id] + agent.fuel #Fuel cost
            agent.evo_net.fitness_evals[teams[id]].append(ig_reward) #Assign those rewards to the members of the team across the sub-populations
        for id, agent in enumerate(gridworld.agent_list_service_bot):
            index = id+parameters.num_agents_scout #service_bot are listed after scouts in teams and rewards vector
            ig_reward = rewards[index]
            #if parameters.is_fuel: ig_reward = rewards[index] + agent.fuel #Fuel cost
            agent.evo_net.fitness_evals[teams[index]].append(ig_reward) #Assign those rewards to the members of the team across the sub-populations

    #Update epoch's best team and save them
    gridworld.epoch_best_team = epoch_best_team
    if generation % 25 == 0:
        is_save = True
        try:
            is_save = (best_global > (tracker.tr_avg_fit[-1][1] - 0.1))
        except: 1+1
        if is_save: gridworld.save_best_team(generation)

    if parameters.use_hall_of_fame: #HAll of Fame
        update_whole_team_hof = True
        if gridworld.agent_list[0].evo_net.hof_net != None: #Quit first time (special case for first run)
            for sub_pop, agent in enumerate(gridworld.agent_list):  # For each agent population
                for index in range(len(gridworld.agent_list[sub_pop].evo_net.genome_list)): #Each individual in an agent sub-population
                    rewards, global_reward = hof_util.agent_simulation(gridworld, parameters, sub_pop, index)
                    agent.evo_net.hof_fitness_evals[index].append(rewards[sub_pop]) #Assign hof scores
                    if global_reward > best_global:
                        update_whole_team_hof = False #Don't update the entirety of hof
                        best_global = global_reward;  # Store the best global performance
                        if best_global > best_hof_score:
                            gridworld.agent_list[sub_pop].evo_net.hof_net = deepcopy(gridworld.agent_list[sub_pop].evo_net.net_list[int(index)]) #Update the HOF team
                            update_whole_team_hof = False
                            print 'HOF CHANGE', global_reward

    #Update entire hall of fame team (only if better team found and that team did not include previous Hall of Famers)
    #TODO testing
    if parameters.use_hall_of_fame and best_global > best_hof_score and update_whole_team_hof: #Hall of fame-new team found
        for i in range(len(best_team)):
            gridworld.agent_list[i].evo_net.hof_net = deepcopy(gridworld.agent_list[i].evo_net.net_list[int(best_team[i])])
            #gridworld.agent_list[i].evo_net.hof_net = deepcopy(nn.create_feed_forward_phenotype(gridworld.agent_list[i].evo_net.genome_list[best_team[i]]))
            print 'HOF changed', best_global, gridworld.agent_list[i].evo_net.hof_net


    for agent in gridworld.agent_list_scout:
        agent.evo_net.update_fitness()# Assign fitness to genomes #
        agent.evo_net.epoch() #Run Epoch update in the population
        if parameters.baldwin and parameters.update_sim and not parameters.share_sim_subpop: agent.evo_net.bald.port_best_sim()  # Deep copy best simulator
    for agent in gridworld.agent_list_service_bot:
        agent.evo_net.update_fitness()# Assign fitness to genomes #
        agent.evo_net.epoch() #Run Epoch update in the population
        if parameters.baldwin and parameters.update_sim and not parameters.share_sim_subpop: agent.evo_net.bald.port_best_sim()  # Deep copy best simulator


    return best_global

def run_simulation(parameters, gridworld, teams, is_test = False): #Run simulation given a team and return fitness for each individuals in that team

    if is_test: trajectory_log = []
    gridworld.reset(teams)  # Reset board
    #mod.dispGrid(gridworld)
    if parameters.num_agent_failure != 0:
        scout_fails = np.random.choice(parameters.num_agents_scout, parameters.num_agent_failure, replace=False)
        service_fails = np.random.choice(parameters.num_agents_service_bot, parameters.num_agent_failure * 2, replace=False)

    for steps in range(parameters.total_steps):  # One training episode till goal is not reached
        if is_test: ig_traj_log = []

        for id, agent in enumerate(gridworld.agent_list_scout):  #get all the action choices from the agents
            if parameters.num_agent_failure != 0:
                if id in scout_fails: continue
            if steps == 0: agent.perceived_state = gridworld.get_state(agent, is_Scout=True) #Update all agent's perceived state
            if steps == 0 and parameters.split_learner: agent.split_learner_state = gridworld.get_state(agent, is_Scout= True, state_representation=2) #If split learner
            if is_test:
                agent.take_action_test()
                ig_traj_log.append(agent.position[0])
                ig_traj_log.append(agent.position[1])
                #print agent.position
            else:
                agent.take_action(teams[id]) #Make the agent take action using the Evo-net with given id from the population

        for id, agent in enumerate(gridworld.agent_list_service_bot):  #get all the action choices from the agents
            if parameters.num_agent_failure != 0:
                if id in service_fails: continue
            if steps == 0: agent.perceived_state = gridworld.get_state(agent, is_Scout=False) #Update all agent's perceived state
            if steps == 0 and parameters.split_learner: agent.split_learner_state = gridworld.get_state(agent, is_Scout= False, state_representation=2) #If split learner
            if is_test:
                agent.take_action_test()
                ig_traj_log.append(agent.position[0])
                ig_traj_log.append(agent.position[1])
            else:
                agent.take_action(teams[id+parameters.num_agents_scout]) #Make the agent take action using the Evo-net with given id from the population

        if is_test:
            for poi in gridworld.poi_list:
                ig_traj_log.append(poi.position[0])
                ig_traj_log.append(poi.position[1])
            trajectory_log.append(np.array(ig_traj_log))
        gridworld.move() #Move gridworld

        #mod.dispGrid(gridworld)
        #raw_input('E')
        gridworld.update_poi_observations() #Figure out the POI observations and store all credit information
        if parameters.is_poi_move: gridworld.poi_move()

        for id, agent in enumerate(gridworld.agent_list_scout): agent.referesh(teams[id], gridworld) #Update state and learn if applicable
        for id, agent in enumerate(gridworld.agent_list_service_bot): agent.referesh(teams[id],
                                                                         gridworld)  # Update state and learn if applicable

        if gridworld.check_goal_complete(): break #If all POI's observed
        #mod.dispGrid(gridworld)

    #Log final position
    if is_test:
        ig_traj_log = []
        for agent in gridworld.agent_list_scout:
            ig_traj_log.append(agent.position[0])
            ig_traj_log.append(agent.position[1])

        for agent in gridworld.agent_list_service_bot:
            ig_traj_log.append(agent.position[0])
            ig_traj_log.append(agent.position[1])

        for poi in gridworld.poi_list:
            ig_traj_log.append(poi.position[0])
            ig_traj_log.append(poi.position[1])
        trajectory_log.append(np.array(ig_traj_log))






    rewards, global_reward = gridworld.get_reward(teams)
    #print rewards, global_reward
    #print rewards
    #rewards -= 0.001 * steps #Time penalty

    if is_test:
        #trajectory_log = np.array(trajectory_log)
        return rewards, global_reward, trajectory_log

    return rewards, global_reward

def random_baseline():
    total_trials = 10
    g_reward = 0.0
    for trials in range(total_trials):
        best_reward = 0
        for iii in range(population_size*5):
            nn_state, steps, tot_reward = reset_board()  # Reset board
            for steps in range(total_steps):  # One training episode till goal is not reached
                all_actions = []  # All action choices from the agents
                for agent_id in range(num_agents):  # get all the action choices from the agents
                    action = randint(0,4); all_actions.append(action)  # Store all agent's actions
                gridworld.move(all_actions)  # Move gridworld
                gridworld.update_poi_observations()  # Figure out the POI observations and store all credit information

                # Get new nnstates after all an episode of moves have completed
                for agent_id in range(num_agents):
                    nn_state[agent_id] = gridworld.get_first_state(agent_id, use_rnn, sensor_avg, state_representation)
                if gridworld.check_goal_complete(): break

            reward = 100 * sum(gridworld.goal_complete) / num_poi  # Global Reward
            if reward > best_reward: best_reward = reward
            #End of one full population worth of trial
        g_reward += best_reward

    print 'Random Baseline: ', g_reward/total_trials

if __name__ == "__main__":

    if gen_traj:
        test_policies()
        sys.exit()
    #random_baseline()
    mod.dispGrid(gridworld)

    best_hof_score = 0

    for gen in range (parameters.total_generations): #Main Loop
        curtime = time.time()
        best_global = evolve(gridworld, parameters, gen, best_hof_score) #CCEA
        if best_global > best_hof_score: best_hof_score = best_global
        tracker.add_fitness(best_global, gen) #Add best global performance to tracker

        #if parameters.use_neat and not parameters.use_py_neat: tracker.add_mpc(gridworld, parameters) #Update mpc statistics
        elapsed = time.time() - curtime

        if parameters.use_hall_of_fame:
            for i in range(1):
                hof_team_rewards, hof_team_score = hof_util.agent_simulation(gridworld, parameters)
                print hof_team_rewards, hof_team_score
        #for agent in gridworld.agent_list: print agent.evo_net.hof_net
        #continue
        if parameters.use_neat and not parameters.use_py_neat :
            print 'Gen:', gen, ' D' if parameters.D_reward else ' G',  ' Best g_reward', int(best_global * 100), ' Avg:', int(100 * tracker.avg_fitness), '  BEST HOF SCORE: ', best_hof_score,  '  Fuel:', 'ON' if parameters.is_fuel else 'Off' , '  Time_offset type:', 'Hard' if parameters.is_hard_time_offset else 'Soft', 'Time_offset: ', parameters.time_offset #, 'Delta MPC:', int(tracker.avg_mpc), '+-', int(tracker.mpc_std), 'Elapsed Time: ', elapsed #' Delta generations Survival: '      #for i in range(num_agents): print all_pop[i].delta_age / params.PopulationSize,

        else:
            print 'Gen:', gen, 'Reward shaping: ',' Difference   ' if parameters.D_reward else ' Global   ', ' Best global', int(best_global * 100), ' Avg:', int(100 * tracker.avg_fitness), 'Best hof_score: ', best_hof_score
            #for i in range(num_agents): print all_pop[i].pop.longest_survivor,
            #print













