import numpy as np
import matplotlib.pyplot as plt
from networkx import nx
import seaborn as sns
from scipy.stats import truncnorm
import collections
import copy
import itertools



def Initial_options(Number_of_options, Number_of_citizens, t):
    np.random.seed(t)
    optimal_options = np.random.uniform(size = Number_of_options) 
    return optimal_options

def Initial_citizens(Number_of_options, Number_of_citizens, std_initial_citizens,lower_b,upper_b):
    citizen_mat = np.zeros((Number_of_citizens,9))
    citizen_mat[:, 0] = np.random.randint(Number_of_options, size=Number_of_citizens)
    gas_rand = lambda x: truncnorm.rvs((lower_b - optimal_options[int(x)]) / std_initial_citizens,
                                       (upper_b - optimal_options[int(x)]) / std_initial_citizens,
                                       loc=optimal_options[int(x)],scale=std_initial_citizens)
    vfunc = np.vectorize(gas_rand)
    citizen_mat[:, 1] = vfunc(citizen_mat[:,0])
    citizen_mat[:, 2] = citizen_mat[:, 1]
    # Citizen mat is a matrix of:
    # [0] #NO. of option [1] initial estimated value [2] final value [3] cost 
    # [4] decided?(1/0)  [5] distance while made a decision [6] count human meetings
    # [7] Used Normal WOM [8] Used SH WOM 
    # [9] why decided? (0-normal_WOM, 1-SH_WOM, 2- HA) [10] when decided
    return citizen_mat

def Initial_capacity(citizen_mat, Number_of_options,precent_cap):
    counter_per_opt = collections.Counter(citizen_mat[:, 0])
    capacity_mat = np.matrix([[int((counter_per_opt[i])*precent_cap) for i in range(Number_of_options)], np.zeros(Number_of_options)])
    return capacity_mat    

#def building_network(citizen_mat, p_net): ## Erdos network, completely random ##
#    ''' GNP random over the space '''
#    NET = nx.gnp_random_graph(citizen_mat.shape[0], p_net)
#    return NET

def gnp_random_graph_Renana(n ,rand_num, p_1, p_2, p_12, SH_per, seed=None, directed=False):
    """Returns a random graph with two populations
    Parameters
    ----------
    n : int
        The number of nodes.
    rand_num : float
        Location in the parameter space
    p_1 : float
        Probability for edge creation in the SH group // More correct is the distance among close friends.
    p_2 : float
        Probability for edge creation in the NORMAL group.
    p_12 : float
        Probability for edge creation in between the NORMAL and the SH groups.
    SH_per : float
        The amount of SH in the network
    seed : int, optional
        Seed for random number generator (default=None).
    directed : bool, optional (default=False)
        If ``True``, this function returns a directed graph.

    Notes
    -----
    This algorithm runs in `O(n^2)` time.
    References
    ----------
    .. [1] P. Erdős and A. Rényi, On Random Graphs, Publ. Math. 6, 290 (1959).
    .. [2] E. N. Gilbert, Random Graphs, Ann. Math. Stat., 30, 1141 (1959).
    .. [3] R. Peres, The impact of network characteristics on the diffusion
            of innovations, Physica A, (2014)
    """
    if directed:
        G=nx.DiGraph()
    else:
        G=nx.Graph()
    G.name="gnp_random_graph_Renana(%s,%s,%s,%s)"%(n,p_1,p_2,p_12)
    
    SH_size = int(SH_per*n)
    G.add_nodes_from(range(SH_size), group ='SH')
    G.add_nodes_from(range(SH_size,n), group ='Normal')
    color_map = {'SH':'#9B0029', 'Normal':'#003366'}
    colors = [color_map[G._node[node]['group']] for node in G]
             
    if not seed is None:
        np.random.seed(seed)

    if G.is_directed():
        edges=itertools.permutations(range(n),2)
    else:
        edges=itertools.combinations(range(n),2)
            
    for e in edges:
        if (e[0] < SH_size) and (e[1] < SH_size):
            if np.abs(rand_num[e[0]] - rand_num[e[1]])< p_1:    
                G.add_edge(*e)
        if (e[0] >= SH_size) and (e[1] >= SH_size):
            if np.abs(rand_num[e[0]] - rand_num[e[1]]) < p_2:
                G.add_edge(*e)            
        if ((e[0] < SH_size) and (e[1] >= SH_size)) or ((e[0] >= SH_size) and (e[1] < SH_size)):
            if np.random.uniform() < p_12:       
                G.add_edge(*e)        
            
    return G, colors

def calculating_degree_avg(n, p_1, p_2, p_12, SH_per):
    SH_size = int(SH_per*n)
    D_1 = p_1*(n-SH_size-1) + p_12*SH_size
    D_2 = p_2*(SH_size-1) + p_12*(n-SH_size)
    return D_1, D_2

def calculating_p(D1,D2,N,N2):
    a = (D1*(N-N2)-N2*D2)/((N-1)*(N-2*N2))
    p12 = (D1-a*(N-N2-1))/N2
    return (a,a,p12)

def building_decision_mat(citizen_mat, capacity_mat, optimal_options, decision_paramater,in_environment = 1):
    distance_mat = np.zeros((citizen_mat.shape[0],len(optimal_options)))
    for i in range(capacity_mat.shape[1]):
        distance_mat[:,i] =  np.abs(citizen_mat[:,2]-optimal_options[i])
    cond_mat = np.matrix(distance_mat) <=  decision_paramater
    distance_mat = np.matrix(distance_mat)  
    # Building matrix only for the citizen who met the condition
    # [:,0] - id, [:,1] - optimal dec, [:,2] - minimal dist
    decision_mat = np.zeros((np.sum(np.sum(cond_mat, axis = 1) == in_environment),3))
    decision_mat[:,0] = np.where((np.sum(cond_mat, axis = 1) == in_environment) == True)[0]
    decision_mat[:,1] = np.array(np.argmax(cond_mat[decision_mat[:,0].astype(int),:], axis = 1))[:,0]    
    decision_mat[:,2] = np.array(np.min(distance_mat[decision_mat[:,0].astype(int),:], axis = 1))[:,0]
    return decision_mat     

def decision_test(citizen_mat, index_cond_d, capacity_mat, optimal_options, decision_paramater):
    ''' The metric of making a decision based on being the only number in the environment that holds the condition  '''
    decision_mat = building_decision_mat(citizen_mat, capacity_mat, optimal_options, decision_paramater, 1)     # Building for everyone decided and not
    for i in range(optimal_options.shape[0]):
        deci_ind_cond = decision_mat[decision_mat[:,1].astype(int) == i,0]
        if sum(decision_mat[:,1].astype(int) == i) > int(capacity_mat[0,i] - capacity_mat[1,i]):
            ## sort and insert the people one by one, by the minimal distance
            Temp_dec_mat = copy.deepcopy(decision_mat[(decision_mat[:,1].astype(int) == i)])
            intersection = np.intersect1d(Temp_dec_mat[:,0].astype(int),index_cond_d)
            True_False_mat = [x in intersection for x in Temp_dec_mat[:,0]]
            Cond_2_mat = Temp_dec_mat[True_False_mat,:]
            Cond_2_mat = Cond_2_mat[Cond_2_mat[:,2].argsort()]
            for ind, j in enumerate(Cond_2_mat[:,0].astype(int)):             
                if ind >= int(capacity_mat[0,i] - capacity_mat[1,i]):
                    break
                else:
                    citizen_mat[j,4] = 1
                    citizen_mat[j,5] = Cond_2_mat[ind,2]
                    capacity_mat[1,i] += 1
        else:
            intersection = np.intersect1d(deci_ind_cond,index_cond_d)
            capacity_mat[1,i] += len(intersection)
            citizen_mat[intersection.astype(int),4] = 1        
    return citizen_mat, capacity_mat
    
def Meet_Agent(citizen_mat, cond, optimal, std, correct,cost, lower_b, upper_b):  
    ''' From gaussian distribution picking a value and "fixing" the citizen preferences '''
    agent_val = np.array([truncnorm.rvs((lower_b - optimal[int(x)]) / std,
                                        (upper_b - optimal[int(x)]) / std, loc=optimal[int(x)],
                                        scale=std) for x in citizen_mat[:,0][cond]])
    citizen_mat[:,2][cond] = (((agent_val - citizen_mat[:,2][cond])/2)*correct) + citizen_mat[:,2][cond]
    citizen_mat[:,3][cond] = citizen_mat[:,3][cond] + cost
    return citizen_mat[cond]

def Meet_WOM(citizen_mat, cond, cost, correct, G, SH_no):
    ''' Metric of meeting - citizen who have decided '''
    ind_decided = np.where(citizen_mat[:,4]==1)
    for i in cond:
        intersection = list(set(G._adj[i].keys()) & set(ind_decided[0]))
        if len(intersection):
            ind_rand_insect = np.random.choice(len(intersection), 1)
            citizen_mat[i,2] = (((citizen_mat[intersection[ind_rand_insect[0]],2] - citizen_mat[i,2])/2)*correct) + citizen_mat[i,2]
            citizen_mat[i,3] = citizen_mat[i,3] + cost
            if intersection[ind_rand_insect[0]] >= SH_no:
                citizen_mat[i,7] = citizen_mat[i,7] + 1 
            else:
                citizen_mat[i,8] = citizen_mat[i,8] + 1 
    return citizen_mat

def Meet_WOM_Met_Ha(citizen_mat, cond, cost, correct, G, SH_no):
    ''' Metric of meeting - citizen who have met Human Agent '''
    ind_met_Ha = np.where(citizen_mat[:,6]>0)
    for i in cond:
        intersection = list(set(G._adj[i].keys()) & set(ind_met_Ha[0]))
        if len(intersection):
            ind_rand_insect = np.random.choice(len(intersection), 1)
            citizen_mat[i,2] = (((citizen_mat[intersection[ind_rand_insect[0]],2] - citizen_mat[i,2])/2)*correct) + citizen_mat[i,2]
            citizen_mat[i,3] = citizen_mat[i,3] + cost
            if intersection[ind_rand_insect[0]] >= SH_no:
                citizen_mat[i,7] = citizen_mat[i,7] + 1 
            else:
                citizen_mat[i,8] = citizen_mat[i,8] + 1 
    return citizen_mat
               
def closest_point(optimal_options, estimated_distance):
    '''' return the index of individual citizen decision and the estimated distance '''
    dist_fun = lambda a, b: abs(a - b)
    vfunc = np.vectorize(dist_fun)
    return np.array([vfunc(optimal_options,estimated_distance).argmin(), vfunc(optimal_options,estimated_distance).min()])

def check_match(optimal_options, citizen_mat):
    ''' return for each citizen if he had made is best decision '''
    decision_vec = np.zeros((citizen_mat.shape[0], 2))
    for ind, estimated_distance in enumerate(citizen_mat[:,2]):
        decision_vec[ind,:] = closest_point(optimal_options, estimated_distance)
    good_prob = (np.sum((decision_vec[:, 0] == citizen_mat[:, 0]).astype(float)))/citizen_mat.shape[0]
    avg_distance = np.mean(decision_vec[:, 1])
    return np.array([good_prob, avg_distance])

def calculate_environment(citizen_mat, capacity_mat, optimal_options, decision_paramater):
    results =np.zeros(6)
    for i in range(3):
         decision_mat = building_decision_mat(citizen_mat, capacity_mat, optimal_options, decision_paramater,i)
         if decision_mat.shape[0] == 0:
             results[i*2:i*2+2]  = [0, 0]
         else:
             results[i*2:i*2+2]  = [decision_mat.shape[0], np.mean(decision_mat[:,2])]
    return results

def Decision_stats(citizen_mat,citizen_mat_before, optimal_options, WOM_decided_mat):
    """ calculate how many people decided by SH and by normal and if it was good decision or not """
    mask_Normal = (citizen_mat[:,4] == 1) & (citizen_mat_before[:,4] == 0) & (citizen_mat[:,7] != citizen_mat_before[:,7])
    WOM_decided_mat[0] = WOM_decided_mat[0] + citizen_mat[mask_Normal].shape[0]
    if citizen_mat[mask_Normal].shape[0]:
        WOM_decided_mat[1] = WOM_decided_mat[1] + check_match(optimal_options, citizen_mat[mask_Normal])[0] * citizen_mat[mask_Normal].shape[0]
    
    mask_SH = (citizen_mat[:,4] == 1) & (citizen_mat_before[:,4] == 0) & (citizen_mat[:,8] != citizen_mat_before[:,8])
    WOM_decided_mat[2] = WOM_decided_mat[2] + citizen_mat[mask_SH].shape[0]
    if citizen_mat[mask_SH].shape[0]:
        WOM_decided_mat[3] = WOM_decided_mat[3] + check_match(optimal_options, citizen_mat[mask_SH])[0] * citizen_mat[mask_SH].shape[0]
    
    mask_Ha = (citizen_mat[:,4] == 1) & (citizen_mat_before[:,4] == 0) & (citizen_mat[:,6] != citizen_mat_before[:,6])
    WOM_decided_mat[4] = WOM_decided_mat[4] + citizen_mat[mask_Ha].shape[0]
    if citizen_mat[mask_Ha].shape[0]:
        WOM_decided_mat[5] = WOM_decided_mat[5] + check_match(optimal_options, citizen_mat[mask_Ha])[0] * citizen_mat[mask_Ha].shape[0]
    
    return WOM_decided_mat

def coverge_time(decided_mat,i):
    if i>15:
        uni_val = np.unique(decided_mat[i-15:i,1])
        if len(uni_val) == 1:
            return i, True
        else:
            return 0, False
    else:
            return 0, False
def run_simulation(citizen_mat, Number_of_citizens,  optimal_options, p_A, std_A, cost_A, correct_A ,p_WOM, cost_WOM, correct_WOM,
                   decision_paramater, threshold, lower_b, upper_b, capacity_mat, i, G, SH_p, p_WOM_Ha):                     
    """ draw a number and make interactions with other agents """
    meet_prob = np.random.uniform(size = (Number_of_citizens))

    agent_cond = [np.intersect1d(np.where(meet_prob<=p_A),
                                 np.where((citizen_mat[:,3] <= (threshold-cost_A-1)) & (citizen_mat[:,4] == 0)))]
    if np.size(agent_cond):   
        citizen_mat[agent_cond] = Meet_Agent(citizen_mat, agent_cond[0], optimal_options, std_A, correct_A,cost_A, lower_b, upper_b)
        citizen_mat[:,6][agent_cond] += 1

    WOM_cond = [np.intersect1d(np.where(((meet_prob<=(p_A  + p_WOM))& (meet_prob>p_A))),
                               np.where((citizen_mat[:,3] <= (threshold-cost_WOM-1)) & (citizen_mat[:,4] == 0)))]

    if np.size(WOM_cond):
        citizen_mat = Meet_WOM(citizen_mat, WOM_cond[0], cost_WOM, correct_WOM, G, SH_p*Number_of_citizens)
        

    WOM_cond_ha = [np.intersect1d(np.where(((meet_prob<=(p_A  + p_WOM + p_WOM_Ha))& (meet_prob>p_A + p_WOM))),
                               np.where((citizen_mat[:,3] <= (threshold-cost_WOM-1)) & (citizen_mat[:,4] == 0)))]

    if np.size(WOM_cond_ha):
        citizen_mat = Meet_WOM_Met_Ha(citizen_mat, WOM_cond_ha[0], cost_WOM, correct_WOM, G, SH_p*Number_of_citizens)
        
    nothing_cond = [np.intersect1d(np.where(meet_prob>=(p_A  + p_WOM + p_WOM_Ha)),
                               np.where((citizen_mat[:,3] <= (threshold-1)) & (citizen_mat[:,4] == 0)))]
    if np.size(nothing_cond):
        citizen_mat[:,3][nothing_cond] += 1

    if np.size(citizen_mat[citizen_mat[:,4] == 0]):
        cond_not_d = [citizen_mat[:,4] == 0]
        index_cond_d = np.where(cond_not_d[0] == True)[0]
        citizen_mat, capacity_mat = decision_test(citizen_mat, index_cond_d, capacity_mat, optimal_options, decision_paramater)
    return citizen_mat,capacity_mat

#####################################    
############# Plots #################
#####################################

def plot_decision_point(optimal_options):
    # Showing the areas of the decisions

    optimal_options.sort()
    avg_vec = np.zeros(len(optimal_options)+1)
    avg_vec[len(optimal_options)] = 1
    for i in range(1, len(optimal_options)+1):
        if i<(len(optimal_options)):
            avg_vec[i] = ((optimal_options[i - 1] + optimal_options[i]) / 2)
        if i%2 == 0:
            ax.axhspan(avg_vec[i-1], avg_vec[i], alpha=0.1, color='red')
        else:
            ax.axhspan(avg_vec[i - 1], avg_vec[i], alpha=0.1, color='blue')
def random_citizen_plot(iter_mat_val):
    plt.title('The change in the decision-making of random citizens')
    plt.xlabel('Iterations')
    rand_citizen = np.random.choice(iter_mat_val.shape[0], 20, replace=False)
    for i in rand_citizen:
        plt.plot(range(iteration), iter_mat_val[i, :],'-.',linewidth = 0.9,alpha=0.8,color='grey')

def plot_density_distribution(citizen_mat, Number_of_options, optimal_options, avg_prob_dist_mat, iteration):
    fig, axes  =  plt.subplots(nrows = 2, ncols = 1, figsize = (6,6))
    
    # Before the simulation
    plt.suptitle('Probability to Optimal decision initial - {0}, and Final - {1}'.format(
            round(avg_prob_dist_mat[0,0],2),round(avg_prob_dist_mat[-1,0],2)))
    
    for i in range(Number_of_options):
        sns.kdeplot(citizen_mat[:,1][citizen_mat[:,0] == i], shade = True, cut = 0, ax = axes[0])
        axes[0].set_xticks([round (opt_round,2) for opt_round in optimal_options], )
        plt.sca(axes[0])
        plt.title('Initial state')
        plt.xticks(optimal_options, [round(e,2) for e in optimal_options], rotation = 90)
        plt.xlim(0,1)
    # After the simulation
    for i in range(Number_of_options):
        sns.kdeplot(citizen_mat[:,2][citizen_mat[:,0] == i], shade = True, cut = 0, ax = axes[1])
        plt.sca(axes[1])
        plt.title('Final state (' +  str(iteration) + ' iter)')
        plt.xticks(optimal_options, [round(e,2) for e in optimal_options], rotation = 90)
        plt.xlim(0,1)
    
    plt.tight_layout(rect = [ 0,0,1,0.95])
    
def plot_prob_and_avg_distance_iter(avg_prob_dist_mat, iteration):
    fig, axes  =  plt.subplots(nrows = 1, ncols = 2, figsize = (10,5))
    axes[0].plot(range(iteration), avg_prob_dist_mat[:,0])
    axes[0].set_title('Probability vs. iteration')
    
    axes[1].plot(range(iteration), avg_prob_dist_mat[:,1])
    axes[1].set_title('Average distance vs. iteration')     

def plot_environment(decided_mat, env_mat,times, decision_paramater):
    fig, axes  =  plt.subplots(nrows = 1, ncols = 2, figsize = (15,6))
    plt.sca(axes[0])
    for i in range(env_mat.shape[1]):
        if (i%2)==0:
            plt.title('AVG status of people vs. iteration \n decision_paramater {0}'.format(round(decision_paramater,3)))
            plt.plot(decided_mat[:,0],env_mat[:,i]/times) 
            plt.ylabel('Number of citizens')           
            plt.xlabel('Iterartions')           
    plt.legend(['0 envi','1 envi (decided)', '2 envi'])
            
    
    for i in range(env_mat.shape[1]):
        plt.sca(axes[1])
        if (i%2)==1:
            plt.title('AVG min distance from option vs. iteration \n decision_paramater {0}'.format(round(decision_paramater,3)))
            plt.plot(decided_mat[:,0],env_mat[:,i]/times)
            plt.ylabel('AVG distance to the closest point')
            plt.xlabel('Iterartions')
            plt.legend(['0 envi','1 envi', '2 envi'])

def Plot_network(G, colors):
    ## Plot graph
    fig, ax = plt.subplots(figsize = (10,5))
    
    nx.draw(G, node_size=50, node_color=colors, width = 0.2, pos=nx.spring_layout(G, k=0.25, iterations=50), alpha = 0.8)
    plt.show()
    
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    
    fig, ax = plt.subplots(figsize = (10,5))
    plt.bar(deg, cnt, width=0.80, color='b')
    
    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    deg_2 = []
    for ind, d in enumerate(deg):
        if ind % 2== 0:
            deg_2.append(d)    
    ax.set_xticks([d for d in deg_2])
    ax.set_xticklabels(deg_2, rotation = 90)
    

#########################################
###### The beginning of the code ########
#########################################

Number_of_citizens = 1000

lower_b = 0
upper_b = 1

Number_of_options= 6
iteration = 250
times = 20
range_t = 1
range_dec = 3
range_WOM = 5
range_WOM_Ha = 5
range_perSH = 5
range_p_A = 5

std_initial_citizens = 1/(Number_of_options+1)
std_A = 1/((Number_of_options)*15)
correct_A = 1
correct_WOM = 0.5
cost_A = 7
cost_WOM = 2
prob_A = np.linspace(0.01,0.1,range_p_A)

p_WOM = np.linspace(0,0.3,range_WOM)
p_WOM_Ha = np.linspace(0,0.3,range_WOM_Ha)

precent_cap = [100]
per_SH = np.linspace(0,0.032,range_perSH)
decision_paramater = np.linspace(3*std_A,12.01*std_A, range_dec)
change_d = [-0.000133333,-0.00033356,-0.000533777]
threshold = np.linspace(200,900,range_t) 

p_1, p_2 = 0.002,0.002
p_12 = 0.2
#D1, D2 = (3.978, 198.018)
D1, D2 = (6, 180)

results_mat = np.zeros((len(change_d)*range_t*range_WOM*range_WOM_Ha*range_dec*range_p_A*times*len(precent_cap)*range_perSH,21))

##D_p_iter_mat = np.zeros((len(change_d)*range_t*range_dec*times*len(precent_cap),iteration+1))

index = 0
for p_A in prob_A:
    # running for different prob_A
    for SH_p in per_SH:
        # running for different Social Hubs percentage
        for w in range(range_WOM):
            # running for different probability to WOM
            for w_ha in range(range_WOM_Ha):
                # running for different probability to WOM_Ha
                for c_p in precent_cap:
                        # running for different capacity
                    for c in range(range_t):
                        # running for different budget
                        for dec in range(range_dec):
                         # running for different decision parameter   
#                         for a in change_d:
#                             # running for different linear coeff of decision paramter decreasing
                             for t in range(times):
                             # running for different initialization states
                                optimal_options = Initial_options(Number_of_options, Number_of_citizens, t)
                                citizen_mat = Initial_citizens(Number_of_options, Number_of_citizens, std_initial_citizens,lower_b,upper_b)
                                p_1, p_2, p_12 = calculating_p(D1,D2,citizen_mat.shape[0],citizen_mat.shape[0]*SH_p)
                                G, colors = gnp_random_graph_Renana(citizen_mat.shape[0], citizen_mat[:,1],p_1,p_2,p_12, SH_p)                    
                                
                                capacity_mat = Initial_capacity(citizen_mat, Number_of_options, c_p)
                                avg_prob_dist_mat = np.zeros((2, 2)) 
                                decided_mat = np.vstack((np.array(range(iteration)), np.zeros(iteration))).T
                                WOM_decided_mat = np.zeros(6)
                                ##env_mat = np.zeros((iteration, 6))
                                ###iter_mat_match = np.zeros((Number_of_citizens,iteration))
                                for i in range(iteration):
                                    if i == 1:
                                        avg_prob_dist_mat[0,:] = check_match(optimal_options, citizen_mat)
                                        prob_just_chosen_initial = check_match(optimal_options, citizen_mat[citizen_mat[:,4] == 1])[0]
                                    elif i == (iteration-1):
                                        avg_prob_dist_mat[1,:] = check_match(optimal_options, citizen_mat)
                                        prob_just_chosen_final = check_match(optimal_options, citizen_mat[citizen_mat[:,4] == 1])[0]
                                    lin_dp = change_d[dec]*i+decision_paramater[dec]
                                    ##env_mat[i,:] = calculate_environment(citizen_mat, capacity_mat, optimal_options, lin_dp) + env_mat[i,:]
                                    decided_mat[i,1] =np.sum(citizen_mat[:,4]) + decided_mat[i,1]
                                    
                                    citizen_mat_before = copy.deepcopy(citizen_mat)                        
                                    citizen_mat, capacity_mat = run_simulation(citizen_mat, Number_of_citizens,  optimal_options, p_A, std_A, cost_A, correct_A ,p_WOM[w],
                                                                               cost_WOM, correct_WOM ,decision_paramater[dec], threshold, lower_b, upper_b, capacity_mat, i, G, SH_p, p_WOM_Ha[w_ha])
                                    
                                    WOM_decided_mat = Decision_stats(citizen_mat,citizen_mat_before, optimal_options, WOM_decided_mat)
                                    
                                    converge_t, converge_cond  = coverge_time(decided_mat,i)
                                    #if converge_cond:
                                        #print(converge_t)
                                    #    break
                                avg_prob_dist_mat[1,:] = check_match(optimal_options, citizen_mat)
                                prob_just_chosen_final = check_match(optimal_options, citizen_mat[citizen_mat[:,4] == 1])[0]        
                                        #break
            
                                ##D_p_iter_mat[index,:] = np.append(a,decided_mat.T[1,:])
                                ########## Meeting 7 #############
                                ##results_mat[index,:] = (decision_paramater[dec], threshold[c], avg_prob_dist_mat[0,0], avg_prob_dist_mat[1,0],prob_just_chosen_initial, prob_just_chosen_final, np.mean(citizen_mat[:,3]),
                                ##           1- decided_mat[1,1]/Number_of_citizens, np.size((citizen_mat[citizen_mat[:,4] == 0]),0)/Number_of_citizens, np.mean(citizen_mat[:,6]),env_mat[0,0]/Number_of_citizens, env_mat[i-1,0]/Number_of_citizens, converge_t, c_p, a, p_WOM[w],
                                ##           np.sum(((citizen_mat[:,7]>0) & (citizen_mat[:,6]==0)))/Number_of_citizens, np.sum(((citizen_mat[:,7]>0) & (citizen_mat[:,6]>0)))/Number_of_citizens)
            
                                results_mat[index,:] = (decision_paramater[dec], threshold[c], avg_prob_dist_mat[0,0], avg_prob_dist_mat[1,0],prob_just_chosen_initial, prob_just_chosen_final, np.mean(citizen_mat[:,3]),
                                           1- decided_mat[1,1]/Number_of_citizens, np.size((citizen_mat[citizen_mat[:,4] == 0]),0)/Number_of_citizens,
                                           np.mean(citizen_mat[:,6]),converge_t, change_d[dec], p_WOM[w],p_WOM_Ha[w_ha] ,SH_p ,WOM_decided_mat[0], WOM_decided_mat[1], WOM_decided_mat[2], WOM_decided_mat[3],
                                           WOM_decided_mat[4], WOM_decided_mat[5])
            
                                index+=1

sns.set()
##plot_environment(decided_mat, env_mat,times, decision_paramater[dec])
### Start plotting
#fig, ax = plt.subplots()
#plot_decision_point(optimal_options)
#random_citizen_plot(iter_mat_val)

plot_density_distribution(citizen_mat, Number_of_options, optimal_options, avg_prob_dist_mat, iteration)
Plot_network(G, colors)
#plot_prob_and_avg_distance_iter(avg_prob_dist_mat, iteration)

