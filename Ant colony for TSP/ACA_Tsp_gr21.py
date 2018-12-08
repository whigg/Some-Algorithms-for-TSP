import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from math import sqrt

#####################################################
# 1. compute distance between every two cities
#  example
n = 21
D = np.zeros((n, n))
t = np.loadtxt('./gr21.txt')
a = []
for i in range(t.shape[0]):
    for j in range(t.shape[1]):
        a.append(t[i][j])
k = 0
for i in range(n):
    for j in range(i+1):
        D[i][j] = a[k]
        D[j][i] = a[k]
        k += 1
for i in range(n):
    D[i][i] = 1e-4
# print(D)
#####################################################
# 2. Initialize all parameters

m = 30  # number of ants
alpha = 1  # Pheromone importance factor
beta = 3  # Heuristic Function Importance Factor
vol = 0.2   # volatilization Factor
Q = 10  # Pheromone constant
Heu_F = 1/D  # Heuristic function
Tau = np.ones((n, n))   # Pheromone matrix
# Path record every ants in one iteration
Table = np.zeros((m, n), dtype=np.int32)    # path of all ants in one iteration
Length = np.zeros((m, 1))
iteration = 0
iter_max = 200  # iter_max

# best route until this iteration
Route_best = np.zeros((iter_max, n), dtype=np.int32)
# length of best route until this iteration
Length_best = np.zeros((iter_max, 1))
Length_ave = np.zeros((iter_max, 1))  # average length of every iteration
Limit_iter = 0  # number of iteration when get the best solution


#####################################################
# 3. iter for best path until get to the iter_max
#   3.1 place ants randomly
#   3.2 For every ants, construct a path
#   3.3 compute the length of every path for every ant
#   3.4 get shortest path(length and path) until this iteration
#       and average length of this interation
#   3.5 update pheromone

# iter for best path
while iteration < iter_max:
    print('%f%%	 iteartion:%d	Length_best:%f' %
          (iteration*100/iter_max, iteration, Length_best[iteration-1]))
    start = random.randint(0, n, size=(m,))
    Table[:, 0] = start

    # for each ant
    for i in range(0, m):
        # for each city
        for j in range(1, n):
            # find the cities that never visited
            visited_city = Table[i, 0:j]
            all_city = np.arange(n, dtype=np.int32)
            allow_city = np.array(
                list(set(all_city).difference(set(visited_city))),
                dtype=np.int32)
            P = np.zeros(allow_city.shape)

            # Computate transfer probability, and next city
            for k in range(len(allow_city)):
                P[k] = pow(Tau[visited_city[-1], allow_city[k]], alpha) * \
                    pow(Heu_F[visited_city[-1], allow_city[k]], beta)

            P = np.array(P/sum(P))

            for k in range(1, len(allow_city)):
                P[k] += P[k - 1]
            P[len(allow_city)-1] = 1.0

            Pc = random.rand()
            target_index = np.where(P > Pc)
            target = allow_city[target_index[0][0]]
            Table[i][j] = target

    # compute path length of every ants
    Length = np.zeros((m, 1))
    for i in range(m):
        Route = Table[i, :]
        for j in range(n-1):
            Length[i] = Length[i] + D[Route[j], Route[j + 1]]

        Length[i] = Length[i] + D[Route[n-1], Route[0]]

    # compute shortest path length until this iteration
    # and average length of this iteration
    if iteration == 0:
        min_Length = np.min(Length)
        min_index = np.where(Length == min_Length)
        # from tuple to single value int
        min_index = min_index[0][0]

        Length_best[iteration] = min_Length
        Length_ave[iteration] = np.mean(Length)
        Route_best[iteration, :] = Table[min_index, :]
        Limit_iter = 1
    else:
        min_Length = np.min(Length)
        min_index = np.where(Length == min_Length)
        # from tuple to single value int
        min_index = min_index[0][0]

        Length_best[iteration] = min(min_Length, Length_best[iteration-1])
        Length_ave[iteration] = np.mean(Length)
        if Length_best[iteration] == min_Length:
            Route_best[iteration, :] = Table[min_index, :]
            if Length_best[iteration] != Length_best[iteration-1]:
                    Limit_iter = iteration
        else:
            Route_best[iteration, :] = Route_best[iteration-1, :]

    # update pheromone
    Delta_Tau = np.zeros((n, n))
    for i in range(m):
        for j in range(n-1):
            Delta_Tau[Table[i][j], Table[i][j+1]] += Q/Length[i]
        Delta_Tau[Table[i][n-1], Table[i][0]] += Q/Length[i]

    Tau = (1-vol)*Tau + Delta_Tau

    # iteration + 1. clear Table
    iteration += 1
    Table = np.zeros((m, n), dtype=np.int32)

#####################################################
# 4. display the path
print('Route_best:', Route_best[iter_max-1])
print('ants number:%d	iter_max:%d	Limit_iter:%d	   best_length:%f \n' %
      (m, iter_max, Limit_iter,	Length_best[iter_max - 1]))
'''
x = []
y = []
for i in range(n):
    x.append(city[Route_best[iter_max-1, i]][0])
    y.append(city[Route_best[iter_max-1, i]][1])
x.append(city[Route_best[iter_max-1, 0]][0])
y.append(city[Route_best[iter_max-1, 0]][1])

plt.plot(x, y, color='b', linewidth=2, alpha=0.5)
plt.scatter(x, y, color='r', s=25)
plt.title('Route_Best')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.show()
'''
