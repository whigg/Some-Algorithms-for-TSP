import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from math import sqrt


class TSP(object):
    def __init__(self):
        self.liveCount = 100
        self.iter_max = 500
        self.initCity()
        self.ga = GA(aCrossRate=1,
                     aMutationRate=0,
                     aLifeCount=self.liveCount,
                     aGeneLength=self.cityNumber,
                     aMatchFun=self.matchFun())

    # 读取城市数据， 计算城市之间的距离
    def initCity(self):
        self.city = np.array([
            [0, 0],
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 0],
            [1, 1],
            [1, 2],
            [1, 3]
        ])
        self.cityNumber = 8
        self.D = np.zeros((self.cityNumber, self.cityNumber))
        self.compute_dis()

    def compute_dis(self):
        for i in range(0, self.cityNumber):
            for j in range(0, self.cityNumber):
                if i != j:
                    temp = (self.city[i, :] - self.city[j, :])
                    self.D[i][j] = ((sqrt(np.square(temp).sum())))
                else:
                    self.D[i][j] = 0.0001

    # 设计适应度函数
    def matchFun(self):
        return lambda life: self.getFitness(life)

    def getFitness(self, life):
        dis = 0.0
        for i in range(self.cityNumber-1):
            dis += self.D[life.gene[i]][life.gene[i+1]]
        dis += self.D[life.gene[self.cityNumber - 1]][life.gene[0]]
        life.distance = dis
        return self.cityNumber**4/dis  # np.exp(self.cityNumber**3/dis)

    # def display(self):
    #     x = []
    #     y = []
    #     print('最优路 :     ', self.ga.best.gene)
    #     print('种群规模:%d	最大迭代次数:%d	程序收敛时迭代次数:%d\n最短路长度:%.2f\n' %(self.ga.lifeCount, self.iter_max,self.ga.Limit_iter, self.ga.best.distance)

    #     for i in range(n):
    #         x.append(city[self.ga.best.gene[i]][0])
    #         y.append(city[self.ga.best.gene[i]][1])
    #     x.append(city[self.ga.best.gene[0]][0])
    #     y.append(city[self.ga.best.gene[0]][1])

    #     plt.figure()
    #     plt.plot(x, y, color='b', linewidth=2, alpha=0.5, label='direct path')
    #     plt.scatter(x, y, color='r', s=10, label='position')
    #     plt.grid(linestyle='-.')
    #     plt.legend()
    #     plt.title('Best Route when m = %d  iter_max = %d  shortest length = %d' %
    #             (m, iter_max, Length_best[iter_max-1]))
    #     plt.xlabel('x value')
    #     plt.ylabel('y value')

    #     x=[i for i in range(self.iter_max)]
    #     y=[Length_best[i] for i in range(self.iter_max)]

    #     plt.figure()
    #     plt.plot(x, y, color='b', linewidth=2,
    #              alpha=0.5, label='Convergence Curve')
    #     plt.scatter(x, y, color='r', s=3)
    #     plt.grid(linestyle='-.')
    #     plt.legend()
    #     plt.title('Convergence Curve when m = %d  iter_max = %d  shortest length = %d' % (
    #         m, iter_max, Length_best[iter_max-1]))
    #     plt.xlabel('Iteartion')
    #     plt.ylabel('Shortest length until this iteartion')

    #     plt.show()

    # 运行TSP问题

    def run(self):
        self.initCity()
        self.ga.initPopulation()
        for _ in range(self.iter_max):
            self.ga.selectNext()
            self.ga.cross()
            self.ga.mutation()
            self.ga.generation += 1
        self.ga.judge()
        print('best_path:', self.ga.best.gene)
        print('shortest_distance:%.2f, best_fitness: %.2f' % (self.ga.best.distance, self.ga.best.fitness))


# 个体类
class Life(object):
    def __init__(self, aGene=None):
        self.gene = aGene
        self.fitness = 0.0
        self.distance = -1.


# 遗传算法类
# 初始化交叉概率、突变概率、种群大小、基因长度、适应度函数
# object：种群、最优秀个体、种群适应度总和
#         当前迭代次数、截至目前总交叉次数、总突变次数
class GA(object):
    def __init__(self, aCrossRate, aMutationRate, aLifeCount, aGeneLength, aMatchFun=lambda life: 1):
        # 需要外部参数传入的属性
        self.crossRate = aCrossRate
        self.mutationRate = aMutationRate
        self.lifeCount = aLifeCount
        self.geneLength = aGeneLength
        self.matchFun = aMatchFun
        self.Limit_iter = 1
        self.shortest_dis_per_generarion = np.zeros((1000,))

        # 内部属性
        self.lives = []
        self.best = None
        self.totalFitness = 0.0  # 适配值总和

        # 内部属性，如果改成私有属性__
        self.generation = 0
        self.crossCount = 0
        self.mutationCount = 0

    # 初始化种群
    def initPopulation(self):
        self.lives = []
        for _ in range(self.lifeCount):
            gene = np.arange(self.geneLength)
            np.random.shuffle(gene)
            life = Life(gene)
            self.lives.append(life)

    # 计算种群适应度
    def judge(self):
        self.totalFitness = 0.0
        self.best = self.lives[0]

        for life in self.lives:
            life.fitness = self.matchFun(life)
            self.totalFitness += life.fitness
            if self.best.fitness < life.fitness:
                self.best = life
                self.Limit_iter = self.generation

    # 从liveCount旧个体里面轮盘赌选择法选择liveCount新个体
    def selectNext(self):
        self.judge()
        self.shortest_dis_per_generarion[self.generation] = self.best.distance
        newLives = []
        P = np.zeros((self.lifeCount,))
        # 此处没有应用精英保留策略
        for i in range(self.lifeCount):
            P[i] = self.lives[i].fitness/self.totalFitness

        for k in range(1, self.lifeCount):
            P[k] += P[k-1]
        P[self.lifeCount-1] = 1.0

        for i in range(self.lifeCount):
            Pc = random.rand()
            target_index = np.where(P > Pc)
            target = target_index[0][0]

            new = Life()
            new.gene = self.lives[target].gene.copy()
            new.fitness = self.lives[target].fitness
            new.distance = self.lives[target].distance

            newLives.append(new)

        self.lives = newLives
        self.generation += 1

    # 在种群内部，反复随机选择两个个体进行交叉操作， 新个体代替旧个体， 以维持liveCount不变

    def cross(self):
        for i in range(int(self.lifeCount*0.1+1)):
            # 随机选择2个parent进行两点交叉
            pc = random.rand()
            if pc < self.crossRate:
                parent1_index = random.randint(0, self.lifeCount)
                parent2_index = random.randint(0, self.lifeCount)
                while parent1_index == parent2_index:
                    parent2_index = random.randint(0, self.lifeCount)

                parent1 = self.lives[parent1_index]
                parent2 = self.lives[parent2_index]

                index1 = random.randint(1, self.geneLength-1)
                index2 = random.randint(index1, self.geneLength)

                temp = parent1.gene.copy()[index1:index2+1]
                parent1.gene[index1:index2+1] = parent2.gene[index1:index2+1]
                parent2.gene[index1:index2+1] = temp

                # 冲突解决
                g1 = list(
                    set(parent1.gene[index1:index2+1]).difference(set(parent2.gene[index1:index2+1])))
                g2 = list(
                    set(parent2.gene[index1:index2+1]).difference(set(parent1.gene[index1:index2+1])))

                for i, city in enumerate(g1):
                    target_index = np.where(parent1.gene == city)
                    if index1 <= target_index[0][0] <= index2:
                        parent1.gene[target_index[0][1]] = g2[i]
                    else:
                        parent1.gene[target_index[0][0]] = g2[i]

                for i, city in enumerate(g2):
                    target_index = np.where(parent2.gene == city)
                    if index1 <= target_index[0][0] <= index2:
                        parent2.gene[target_index[0][1]] = g1[i]
                    else:
                        parent2.gene[target_index[0][0]] = g1[i]

                # 旧个体被新个体代替
                self.lives[parent1_index] = parent1
                self.lives[parent2_index] = parent2

                self.crossCount += 1

        self.judge()

    # 对每个个体的每个基因实施突变
    def mutation(self):
        for live in self.lives:
            pc = random.rand()
            if pc < self.mutationRate:
                index1 = random.randint(0, self.geneLength)
                index2 = random.randint(0, self.geneLength)
                while index1 == index2:
                    index2 = random.randint(0, self.geneLength)

                live.gene[index1], live.gene[index2] = live.gene[index2], live.gene[index1]

                self.mutationRate += 1

        self.judge()


def main():
    tsp = TSP()
    tsp.run()


if __name__ == '__main__':
    main()
