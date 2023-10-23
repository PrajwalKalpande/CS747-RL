"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

You need to complete the following methods:
    - give_pull(self): This method is called when the algorithm needs to
        select the arms to pull for the next round. The method should return
        two arrays: the first array should contain the indices of the arms
        that need to be pulled, and the second array should contain how many
        times each arm needs to be pulled. For example, if the method returns
        ([0, 1], [2, 3]), then the first arm should be pulled 2 times, and the
        second arm should be pulled 3 times. Note that the sum of values in
        the second array should be equal to the batch size of the bandit.
    
    - get_reward(self, arm_rewards): This method is called just after the
        give_pull method. The method should update the algorithm's internal
        state based on the rewards that were received. arm_rewards is a dictionary
        from arm_indices to a list of rewards received. For example, if the
        give_pull method returned ([0, 1], [2, 3]), then arm_rewards will be
        {0: [r1, r2], 1: [r3, r4, r5]}. (r1 to r5 are each either 0 or 1.)
"""

from audioop import reverse
import numpy as np
# START EDITING HERE
import math
from cmath import log
# END EDITING HERE

class AlgorithmBatched:
    def __init__(self, num_arms, horizon, batch_size):
        self.num_arms = num_arms
        self.horizon = horizon
        self.batch_size = batch_size
        self._successes = np.zeros(num_arms)
        self._failures = np.zeros(num_arms)
        assert self.horizon % self.batch_size == 0, "Horizon must be a multiple of batch size"
        # START EDITING HERE
        self.time = 0
        self.sampling_count = 27
        self.k = 8
        self.epsilon =0.75
        self.total = 0
        self.limit = (self.horizon//(self.batch_size))

        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        self.time+=1
        n = self.num_arms
        
        if(self.time>=self.epsilon*self.limit):
            self.k = np.random.randint(low=3,high=8)
        
        probs = np.zeros((self.sampling_count,2))
        arms = np.zeros(self.k,dtype=int)
        pulls = np.zeros(self.k,dtype=int)
        
        for i in range(0,self.sampling_count):
                x = np.random.beta(self._successes+1,self._failures+1)
                probs[i]= [-np.max(x),np.argmax(x)]

        probs = np.sort(probs,axis=0)
        totalprob = sum(prob for prob,index in probs)
        self.total =  0
        for i in range(0,self.k):
                prob = probs[i][0]
                self.total+=int((prob/totalprob)*self.batch_size)
                pulls[i] =int((prob/totalprob)*self.batch_size)
                arms[i]= int(probs[i][1])
 
        if (self.total<self.batch_size):
            n = pulls.size
            pulls[n-1]+=(self.batch_size-self.total)
        return arms,pulls
        # END EDITING HERE
    
    def get_reward(self, arm_rewards):
        # START EDITING HERE
        for index,r in arm_rewards.items():
            for reward in r :
                if(reward):
                    self._successes[index]+=1
                else :
                    self._failures[index]+=1

        # END EDITING HERE