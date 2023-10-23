"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

from cmath import log
import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE
def kl(p,q):
    return (p*log((p/(q+1e-9))+1e-9) + (1-p+1e-9)*(np.log(((1-p)/(1-q+1e-9))+1e-9)+1e-9) + 1e-9).real
# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.__counts = np.zeros(num_arms)
        self.__means = np.zeros(num_arms)
        self.__t = 0
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        self.__t+=1
        if(self.__t<=self.num_arms):
            return self.__t%self.num_arms
        ucb = self.__means + np.sqrt(2*log(self.__t)/(self.__counts+1e-8))
        return np.argmax(ucb)
     
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.__counts[arm_index]+=1
        n = self.__counts[arm_index]
        mean = self.__means[arm_index]
        new_mean = ((n - 1) / n) * mean + (1 / n) * reward
        self.__means[arm_index] = new_mean
        # END EDITING HERE

class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.__means= np.zeros(num_arms)
        self.__counts = np.zeros(num_arms)
        self.__klucb = np.zeros(num_arms)
        self.c = 3
        self.__t=0
        self._mid = 0
        # END EDITING HERE
    
  
    def give_pull(self):
        # START EDITING HERE
        self.__t+=1
        target = (log(self.__t+1e-8) + self.c*log(log(self.__t)+1e-8)).real/(self.__counts+1e-8)
         
        for i in range(self.num_arms):
            low = self.__means[i]
            high = 1.0
            for j in range(9):
                self._mid = (low+high)/2.0
                if(kl(self.__means[i],self._mid)<=target[i]):
                    low = self._mid
                else:
                    high = self._mid
            self.__klucb[i] = self._mid
        return np.argmax(self.__klucb)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.__counts[arm_index]+=1
        n = self.__counts[arm_index]
        mean = self.__means[arm_index]
        new_mean = ((n - 1) / n) * mean + (1 / n) * reward
        self.__means[arm_index] = new_mean
        # END EDITING HERE


class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.__successes = np.zeros(num_arms)
        self.__failures = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        x = np.random.beta(self.__successes+1,self.__failures+1)
        return np.argmax(x)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        if(reward):
            self.__successes[arm_index]+=1
        else:
            self.__failures[arm_index]+=1
        # END EDITING HERE
