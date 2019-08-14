# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem): #all algorithms exactly copy the pseudocode with the addition of the visited set
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    #create blank unordered set of visited nodes
    visited = set()

    #create fringe LIFO stack and insert start state
    fringe = util.Stack()
    fringe.push((problem.getStartState(), [])) #insert start state coords + blank action list into stack

    '''
    #TEST CODE
    #print "Start: ", path.pop()
    state, new_action = path.pop()
    print "State and Action: ", state
    print " ", action
    '''

    #DFS west first of max(4) successors
    while not fringe.isEmpty(): #terminates if no path exists or if path is found
        state, action_list = fringe.pop() #gets in order [West, East, South, North]

        if state in visited:
            continue #move on to next line

        visited.add(state)

        if problem.isGoalState(state):
            return action_list

        #put [WEST EAST SOUTH NORTH] on stack-> [W E S N ...]
        for successor_state, successor_action, stepCost in problem.getSuccessors(state):
            if successor_state not in visited:
                #push state + add to action list for successor
                fringe.push((successor_state, action_list + [successor_action]))

def breadthFirstSearch(problem): #same as DFS except use FIFO queue instead of FIFO stack
    """Search the shallowest nodes in the search tree first."""
    
    #create blank unordered set of visited nodes
    visited = set()

    #create fringe FIFO queue and isnert start state
    fringe = util.Queue()
    fringe.push((problem.getStartState(), []))

    while not fringe.isEmpty(): #if fringe is empty then return failure
        state, action_list = fringe.pop() #remove front

        if state in visited: #prevent repeated computations
            continue

        visited.add(state) #add to set of visited nodes

        if problem.isGoalState(state): #goal test
            return action_list

        for successor_state, successor_action, stepCost in problem.getSuccessors(state): #expand state
            if successor_state not in visited: #prevent extra computations
                fringe.push((successor_state, action_list + [successor_action])) #insert successor into fringe

def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    #create blank unordered set of visited states
    visited = set()

    #create fringe using heap based priority queue and insert start state
    fringe = util.PriorityQueue()
    fringe.push((problem.getStartState(), []), 0) #tuple (state, action, total cost) and cost

    while not fringe.isEmpty():
        state, action_list = fringe.pop()

        if state in visited:
            continue

        visited.add(state)

        if problem.isGoalState(state):
            return action_list

        for successor_state, successor_action, stepCost in problem.getSuccessors(state):
            if successor_state not in visited:
                fringe.push((successor_state, action_list + [successor_action]), stepCost + problem.getCostOfActions(action_list))

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic): #nullHeuristic is default
    """Search the node that has the lowest combined cost and heuristic first."""
    
    #create blank unordered set of visited states
    visited = set()

    #create fringe using heap based priority queue and insert start state
    fringe = util.PriorityQueue()
    fringe.push((problem.getStartState(), []), 0) #tuple (state, action, total cost) and cost

    while not fringe.isEmpty():
        state, action_list = fringe.pop()

        if state in visited:
            continue

        visited.add(state)

        if problem.isGoalState(state):
            return action_list

        for successor_state, successor_action, stepCost in problem.getSuccessors(state):
            if successor_state not in visited:
                fringe.push((successor_state, action_list + [successor_action]), heuristic(successor_state, problem) + stepCost + problem.getCostOfActions(action_list))

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
