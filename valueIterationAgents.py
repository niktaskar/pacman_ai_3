# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations
    self.values = util.Counter() # A Counter is a dict with default 0
     
    "*** YOUR CODE HERE ***"
    # Iterate through the set number of rounds
    # Go through and set values of each state in the dictionary
    possibleStates = self.mdp.getStates()
    new_policy = self.values
    # Go through the iterations
    for i in range(iterations):
      # Set new policy to be remembered to new dictionary
      new_policy = util.Counter()
      # For state in all possible states
      for state in possibleStates:
        # Get the policy for a specific state
        policy = self.getAction(state)
        # If the policy is not None
        if policy is not None:
          new_policy[state] = self.getQValue(state, policy)
        else:
          new_policy[state] = self.values[state]
      self.values = new_policy
    
  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """

    # Returns the value of the given state
    return self.values[state]


  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    "*** YOUR CODE HERE ***"

    # Returns the Q value of the state, action pair
    # Should be the expected "outcome" of all possible resulting states + reward in current state
    ret = 0
    reward = 0
    # Given a state and action
    # We get the transition function and the states that we could enter
    stateProb = self.mdp.getTransitionStatesAndProbs(state, action)
    for s_prime, prob in stateProb:
      # We calculate Q values using the formula below
      reward += prob*self.mdp.getReward(state, action, s_prime)
      discount = self.discount
      v_star = self.getValue(s_prime)
      # Q(s, a) = SUM_(T(s,a,s')[R(s,a,s') + Y*V(s')])
      ret += (prob*(discount*v_star))

    ret += reward
    return ret

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """
    "*** YOUR CODE HERE ***"

    # Returns the best action based on computed values of possible actions in state
    maxVal = float("-inf")
    currMove = None
    possibleActions = self.mdp.getPossibleActions(state)
    if self.mdp.isTerminal(state) or len(possibleActions) == 0:
      return None

    for action in possibleActions:
      # Find the Q(s, a) of all possible actions at a given state
      qVal = self.getQValue(state, action)
      # If the value of the qval for state, action pair is greater than previous thought then we sub
      if qVal > maxVal:
        maxVal = qVal
        currMove = action

    return currMove
    # return self.values[state]

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
  
