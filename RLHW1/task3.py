"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the AlgorithmManyArms class. Here are the method details:
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
"""

import numpy as np

# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE

class AlgorithmManyArms:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.visited = np.zeros(num_arms)
        self.threshold = 0.7
        self.epsilon = 0.1
        self.means = np.zeros(num_arms)
        self.counts = np.zeros(num_arms)
        self.time =0 
        self.previndex = -1
        self.prevresult = 0 
        self.zeros = 0
      

        # Horizon is same as number of arms
    
    def give_pull(self):
        # START EDITING HERE
        self.time+=1
        #01 strategy 

        if(self.time<=self.epsilon*self.num_arms):
            
            index = 0
            if(self.previndex==-1 or self.prevresult==0):
                index = np.random.randint(low =0,high =self.num_arms)
                self.previndex=index
            
            else:
                index = self.previndex
            return index 
        else : 
            #eG2 
            
            if(np.max(self.means)>=self.threshold):
                 return np.argmax(self.means)
                
            else :
                index = np.random.randint(low =0,high =self.num_arms)
                return index






    
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index]+=1
        n = self.counts[arm_index]
        mean = self.means[arm_index]
        new_mean = ((n - 1) / n) * mean + (1 / n) * reward
        self.means[arm_index] = new_mean

        self.prevresult = reward
        if(reward==0) :
            self.zeros+=1
        # END EDITING HERE
