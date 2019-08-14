# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """

        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        '''
        print 'Successor: ', successorGameState.getScore()
        print 'new position: ', newPos
        print 'new Food: ', max(newFood)
        print 'newGhostStates: ', min(newGhostStates)
        print 'newScaredTimes: ', max(newScaredTimes)
        '''
        #score
        score = successorGameState.getScore() - currentGameState.getScore()

        #food left
        foodLeft = len(newFood)
        powerLeft = len(successorGameState.getCapsules())

        #minimum food distance
        minfoodDist = 0
        if foodLeft > 0:
          foodDistances = [manhattanDistance(newPos, (x,y)) for x, row in enumerate(newFood) for y, food in enumerate(row) if food]
          minfoodDist = min(foodDistances)

        #minimum distance from ghost
        scared, active = [], []
        for ghost in successorGameState.getGhostStates():
          if ghost.scaredTimer:
            scared.append(ghost)
          else:
            active.append(ghost)

        def getDist(ghosts):
          return map(lambda ghost: manhattanDistance(newPos, ghost.getPosition()), ghosts)

        if active:
          minactiveDist = min(getDist(active))
        else:
          minactiveDist = 0

        if scared:
          minscaredDist = min(getDist(scared))
        else:
          minscaredDist = 0

        return 10*score - 5/(1+foodLeft) - 5*powerLeft - 1.5/(1+minfoodDist) - 10/(.001+minactiveDist) - 2*minscaredDist

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """

        #use pseudocode given in class via nested functions 
        def maxValue(gameState, depth, numGhosts): #gets max value from MAX-NODE
          if gameState.isWin() or gameState.isLose() or depth == 0: #if terminal depth state
            return self.evaluationFunction(gameState)

          maxVal = -(float("inf"))
          legalActions = gameState.getLegalActions(0)
          for action in legalActions: #takes maxvalue of 
            maxVal = max(maxVal, minValue(gameState.generateSuccessor(0, action), depth, 1, numGhosts))

          return maxVal

        def minValue(gameState, depth, agentIndex, numGhosts): #gets min value from MIN-NODE
          if gameState.isWin() or gameState.isLose() or depth == 0: #if terminal depth state
            return self.evaluationFunction(gameState)

          minVal = (float("inf"))
          legalActions = gameState.getLegalActions(agentIndex)
          for action in legalActions:
            if agentIndex == numGhosts:
              minVal = min(minVal, maxValue(gameState.generateSuccessor(agentIndex, action), depth-1, numGhosts))
            else:
              minVal = min(minVal, minValue(gameState.generateSuccessor(agentIndex, action), depth, agentIndex+1, numGhosts))

          return minVal

        #MINIMAX ALGORITHM: value 'function'
        optimalAction = []
        score = -(float("inf")) #for use as IC
        legalActions = gameState.getLegalActions() #get legal actions of current pacman state 
        for action in legalActions:
          lastScore = score
          score = max(lastScore, minValue(gameState.generateSuccessor(0, action), self.depth, 1, gameState.getNumAgents()-1))

          if score > lastScore:
            optimalAction = action

        return optimalAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction

          alpha = MAX's best option to root
          beta = MIN's best option to root

          Using algorithm from: 
            Russell, Stuart J.; Norvig, Peter (2003), 
            Artificial Intelligence: A Modern Approach (2nd ed.), 
            Upper Saddle River, New Jersey: Prentice Hall, ISBN 0-13-790395-2
        """
        #use pseudocode given in class via nested functions 
        def maxValue(gameState, alpha, beta, depth, numGhosts): #gets max value from MAX-NODE
          if gameState.isWin() or gameState.isLose() or depth == 0: #if terminal depth state
            return self.evaluationFunction(gameState)

          maxVal = -(float("inf"))
          for action in gameState.getLegalActions(0):
            maxVal = max(maxVal, minValue(gameState.generateSuccessor(0, action), alpha, beta, depth, 1, numGhosts))
            alpha = max(alpha, maxVal)

            if alpha > beta:
              return maxVal

          return maxVal

        def minValue(gameState, alpha, beta, depth, agentIndex, numGhosts): #gets min value from MIN-NODE
          if gameState.isWin() or gameState.isLose() or depth == 0: #if terminal depth state
            return self.evaluationFunction(gameState)

          minVal = (float("inf"))
          for action in gameState.getLegalActions(agentIndex):
            if agentIndex == numGhosts:
              minVal = min(minVal, maxValue(gameState.generateSuccessor(agentIndex, action), alpha, beta, depth-1, numGhosts))
              beta = min(beta, minVal)

              if alpha > beta:
                return minVal
            else:
              minVal = min(minVal, minValue(gameState.generateSuccessor(agentIndex, action), alpha, beta, depth, agentIndex+1, numGhosts))
              beta = min(beta, minVal)

              if alpha > beta:
                return minVal

          return minVal

        #MINIMAX ALGORITHM: value 'function'
        optimalAction = []
        score = -(float("inf")) #for use as IC
        alpha = -(float("inf"))
        beta = (float("inf"))
        for action in gameState.getLegalActions(): #get legal actions of current pacman state 
          lastScore = score
          score = max(lastScore, minValue(gameState.generateSuccessor(0, action), alpha, beta, self.depth, 1, gameState.getNumAgents()-1))
          alpha = max(alpha, score)

          if score > lastScore:
            optimalAction = action

          if alpha > beta:
            return optimalAction

        return optimalAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """ 
        def maxValue(gameState, depth, numGhosts): #gets max value from MAX-NODE
          if gameState.isWin() or gameState.isLose() or depth == 0: #if terminal depth state
            return self.evaluationFunction(gameState)

          maxVal = -(float("inf"))
          for action in gameState.getLegalActions(0):
            maxVal = max(maxVal, expectedValue(gameState.generateSuccessor(0, action), depth, 1, numGhosts))

          return maxVal

        def expectedValue(gameState, depth, agentIndex, numGhosts):
          if gameState.isWin() or gameState.isLose() or depth == 0: #if terminal depth state
            return self.evaluationFunction(gameState)

          expectedVal = 0
          legalActions = gameState.getLegalActions(agentIndex)
          for action in legalActions:
            if agentIndex == numGhosts:
              expectedVal += maxValue(gameState.generateSuccessor(agentIndex, action), depth-1, numGhosts)
            else:
              expectedVal += expectedValue(gameState.generateSuccessor(agentIndex, action), depth, agentIndex+1, numGhosts)

          return expectedVal / len(legalActions)
        
        optimalAction = []
        score = -(float("inf")) #for use as IC
        for action in gameState.getLegalActions(): #get legal actions of current pacman state 
          lastScore = score
          score = max(lastScore, expectedValue(gameState.generateSuccessor(0, action), self.depth, 1, gameState.getNumAgents()-1))

          if score > lastScore:
            optimalAction = action

        return optimalAction

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: My evaluation function was uilt such that Pacman will always win and after 
      some time of trying, I can't seem to figure out how to get the game to speed up and the 
      score to e positive. For every positive score Pacman gets, there seems to be one that 
      Pacman takes long enough to get a large negative score. The evaluation function works as such:

      Evaluation Function
      ===================

      evaluationfunction = 
                            1 * score                     #takes the score as the base
                            -5 * foodLeft                 #when food decreases the evalfunc increases
                            -2 * minfoodDist              #when minfoodDist decreases the evalfunc increases
                            -2 / (.0001+minactiveDist)    #when the minactiveDist increases the evalfunc decreases.
                            -15 * powerLeft               #when the ppower pellets left decreases, the evalfunc increases
                            -2 * minscaredDist            #when the minscaredDist decreases the evalfunc increases

      Question q5
      ===========

      Pacman emerges victorious! Score: -2886
      Pacman emerges victorious! Score: 662
      Pacman emerges victorious! Score: -3318
      Pacman emerges victorious! Score: -3410
      Pacman emerges victorious! Score: 586
      Pacman emerges victorious! Score: 144
      Pacman emerges victorious! Score: 232
      Pacman emerges victorious! Score: -3556
      Pacman emerges victorious! Score: -1080
      Pacman emerges victorious! Score: 16
      Average Score: -1261.0
      Scores:        -2886.0, 662.0, -3318.0, -3410.0, 586.0, 144.0, 232.0, -3556.0, -1080.0, 16.0
      Win Rate:      10/10 (1.00)
      Record:        Win, Win, Win, Win, Win, Win, Win, Win, Win, Win
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    #score
    score = currentGameState.getScore() - currentGameState.getScore()

    #food left
    foodLeft = len(newFood)
    powerLeft = len(currentGameState.getCapsules())

    #minimum food distance
    minfoodDist = 0
    if foodLeft > 0:
      foodDistances = [manhattanDistance(newPos, (x,y)) for x, row in enumerate(newFood) for y, food in enumerate(row) if food]
      minfoodDist = min(foodDistances)

    #minimum distance from ghost
    scared, active = [], []
    for ghost in currentGameState.getGhostStates():
      if ghost.scaredTimer:
        scared.append(ghost)
      else:
        active.append(ghost)

    def getDist(ghosts):
      return map(lambda ghost: manhattanDistance(newPos, ghost.getPosition()), ghosts)

    if active:
      minactiveDist = min(getDist(active))
    else:
      minactiveDist = 0

    if scared:
      minscaredDist = min(getDist(scared))
    else:
      minscaredDist = 0

    if minactiveDist == 1:
      minactiveDist = 1000

    return score - 5*foodLeft - 2*minfoodDist - 2/(.0001+minactiveDist) - 15*powerLeft - 2*minscaredDist
# Abbreviation
better = betterEvaluationFunction

