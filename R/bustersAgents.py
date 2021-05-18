from __future__ import print_function
import numpy as np
# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from builtins import range
from builtins import object
import util
# from game import Agent
# from game import Directions
import inference
from keyboardAgents import KeyboardAgent
import busters

from distanceCalculator import Distancer
# from game import Actions
# from game import Directions
import random
import sys
import math

# Qlearning
from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *


class NullGraphics(object):
    "Placeholder for graphics"

    def initialize(self, state, isBlue=False):
        pass

    def update(self, state):
        pass

    def pause(self):
        pass

    def draw(self, state):
        pass

    def updateDistributions(self, dist):
        pass

    def finish(self):
        pass


class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """

    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions:
            self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent(object):
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__(self, index=0, inference="ExactInference", ghostAgents=None, observeEnable=True, elapseTimeEnable=True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution()
                             for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + \
            [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        # for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        # self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."
        return Directions.STOP


class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index=0, inference="KeyboardInference", ghostAgents=None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)


'''Random PacMan Agent'''


class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    ''' Example of counting something'''

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food

    ''' Print the layout'''

    def printGrid(self, gameState):
        table = ""
        # print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + \
                    gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0)  # Legal position from the pacman
        move_random = random.randint(0, 3)
        if (move_random == 0) and Directions.WEST in legal:
            move = Directions.WEST
        if (move_random == 1) and Directions.EAST in legal:
            move = Directions.EAST
        if (move_random == 2) and Directions.NORTH in legal:
            move = Directions.NORTH
        if (move_random == 3) and Directions.SOUTH in legal:
            move = Directions.SOUTH
        return move


class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i+1]]
        return Directions.EAST


class BasicAgentAA(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0

        gameState.get

    ''' Example of counting something'''

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if(height == True):
                    food = food + 1
        return food

    ''' Print the layout'''

    def printGrid(self, gameState):
        table = ""
        # print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + \
                    gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions,
              " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ",
              gameState.data.agentStates[0].getDirection())
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ", [gameState.getGhostDirections().get(
            i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ",
              gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        print(gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore())

    def chooseAction(self, gameState):
        self.countActions = self.countActions + 1
        self.printInfo(gameState)
        move = Directions.STOP
        legal = gameState.getLegalActions(0)  # Legal position from the pacman
        move_random = random.randint(0, 3)
        if (move_random == 0) and Directions.WEST in legal:
            move = Directions.WEST
        if (move_random == 1) and Directions.EAST in legal:
            move = Directions.EAST
        if (move_random == 2) and Directions.NORTH in legal:
            move = Directions.NORTH
        if (move_random == 3) and Directions.SOUTH in legal:
            move = Directions.SOUTH
        return move

    def printLineData(self, gameState):
        return "XXXXXXXXXX"


###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################
###################################################################################################

class QLearningAgent(BustersAgent):

    def __init__(self, **args):
        BustersAgent.__init__(self, **args)
        self.actions = {"North": 0, "East": 1, "South": 2, "West": 3}
        self.epsilon = 0.05
        self.alpha = 0.05
        self.discount = 0.05
        self.reward = 0
        self.numGhosts = 4
        self.countActions = 0

        self.states = {"LEFT": 0, "RIGHT": 1, "UP": 2, "DOWN": 3}

        if os.path.isfile('qtable.txt'):
            self.table_file = open("qtable.txt", "r+")
            self.q_table = self.readQtable()
        else:
            self.table_file = open("qtable.txt", "w+")
            self.q_table = np.zeros((len(self.states), len(self.actions)))

    def readQtable(self):
        "Read qtable from disc"
        table = self.table_file.readlines()
        q_table = []

        for i, line in enumerate(table):
            row = line.split()
            row = [float(x) for x in row]
            q_table.append(row)
        return q_table

    def writeQtable(self):
        "Write qtable to disc"
        self.table_file.seek(0)
        self.table_file.truncate()
        for line in self.q_table:
            for item in line:
                self.table_file.write(str(item) + " ")
            self.table_file.write("\n")

    def __del__(self):
        "Destructor. Invokation at the end of each episode"
        self.writeQtable()
        self.table_file.close()

    def getStateData(self, gameState):
        ''' STATES:
                0)  LEFT
                1)  RIGHT
                2)  UP 
                3)  DOWN
                -1) FINAL_STATE
        '''
        move = -1
        try:
            self.countActions = self.countActions + 1

            # Legal position from the pacman
            legal = gameState.getLegalActions(0)
            ghostsDistances = gameState.data.ghostDistances  # Manhattan distance to ghosts

            minGhostsDistance = min(
                x for x in ghostsDistances if x is not None)
            ghostPosition = gameState.getGhostPositions(
            )[ghostsDistances.index(minGhostsDistance)]
            validMovements = gameState.getLegalPacmanActions()
            pacPosition = gameState.getPacmanPosition()
            xDistanceToGhost = abs(ghostPosition[0] - pacPosition[0])
            yDistanceToGhost = abs(ghostPosition[1] - pacPosition[1])
            prevMove = gameState.data.agentStates[0].getDirection()
            x, y = gameState.getPacmanPosition()
            #print("$", prevMove)
            if(len(validMovements) == 5 and not gameState.getWalls()[x+1][y+1] and not gameState.getWalls()[x+1][y-1] and not gameState.getWalls()[x-1][y+1] and not gameState.getWalls()[x-1][y-1]):
                print("LIBRE")
                if(xDistanceToGhost > yDistanceToGhost):

                    if(ghostPosition[0] > pacPosition[0]):
                        move = 1
                    elif(ghostPosition[0] < pacPosition[0]):
                        move = 0
                else:
                    if(ghostPosition[1] > pacPosition[1]):
                        move = 2
                    else:
                        move = 3
            else:
                # EJE X
                if(xDistanceToGhost > yDistanceToGhost):
                    # PREFERENCIA ESTE
                    if(ghostPosition[0] > pacPosition[0]):
                        if(prevMove == Directions.WEST and (not gameState.getWalls()[x][y+1] or not gameState.getWalls()[x][y-1])):
                            print("OBSTACULO NORTE/SUR - BORDEANDO FIN")
                            move_random = random.randint(0, 1)
                            if (move_random == 0):
                                if(Directions.NORTH in legal):
                                    move = 2
                                elif(Directions.SOUTH in legal):
                                    move = 3
                            elif (move_random == 1):
                                if(Directions.SOUTH in legal):
                                    move = 3
                                if(Directions.NORTH in legal):
                                    move = 2
                        elif(prevMove == Directions.NORTH and gameState.getWalls()[x+1][y] and Directions.NORTH in legal):
                            move = 2
                            print("OBSTACULO ESTE - BORDEANDO ARRIBA")
                        elif(prevMove == Directions.SOUTH and gameState.getWalls()[x+1][y] and Directions.SOUTH in legal):
                            move = 3
                            print("OBSTACULO ESTE - BORDEANDO ABAJO")
                        elif(Directions.EAST not in legal):
                            print("OBSTACULO ESTE")
                            up = y
                            down = y
                            while down > 0 and gameState.getWalls()[x+1][down]:
                                down -= 1

                            while up < gameState.data.layout.height and gameState.getWalls()[x+1][up]:
                                up += 1
                            goUp = y - down > up - y

                            if((y-down == 0 or goUp or Directions.SOUTH not in legal) and Directions.NORTH in legal):
                                move = 2
                            elif((up+y == gameState.data.layout.height or not goUp or Directions.NORTH not in legal) and Directions.SOUTH in legal):
                                move = 3
                            elif(Directions.WEST in legal):
                                move = 0
                        else:
                            print("PREFERENCIA ESTE")
                            move = 1
                    # PREFERENCIA OSTE
                    else:
                        if(prevMove == Directions.EAST and (not gameState.getWalls()[x][y+1] or not gameState.getWalls()[x][y-1])):
                            print("OBSTACULO NORTE/SUR - BORDEANDO FIN")
                            move_random = random.randint(0, 1)
                            if (move_random == 0):
                                if(Directions.NORTH in legal):
                                    move = 2
                                elif(Directions.SOUTH in legal):
                                    move = 3
                            elif (move_random == 1):
                                if(Directions.SOUTH in legal):
                                    move = 3
                                if(Directions.NORTH in legal):
                                    move = 2
                        elif(prevMove == Directions.NORTH and gameState.getWalls()[x-1][y] and Directions.NORTH) in legal:
                            move = 2
                            print("OBSTACULO OESTE - BORDEANDO ARRIBA")
                        elif(prevMove == Directions.SOUTH and gameState.getWalls()[x-1][y] and Directions.SOUTH in legal):
                            move = 3
                            print("OBSTACULO OESTE - BORDEANDO ABAJO")
                        elif(Directions.WEST not in legal):
                            print("OBSTACULO OESTE")
                            down = y
                            up = y
                            while down > 0 and gameState.getWalls()[x-1][down]:
                                down -= 1

                            while up < gameState.data.layout.height and gameState.getWalls()[x-1][up]:
                                up += 1
                            goUp = y - down > up - y
                            if((y-down == 0 or goUp or Directions.SOUTH not in legal) and Directions.NORTH in legal):
                                move = 2
                            elif((up+y == gameState.data.layout.height or not goUp or Directions.NORTH not in legal) and Directions.SOUTH in legal):
                                move = 3
                            elif(Directions.EAST in legal):
                                move = 1
                        else:
                            move = 0
                            print("PREFERENCIA OESTE")
                # EJE Y
                else:
                    # PREFERENCIA NORTE
                    if(ghostPosition[1] > pacPosition[1]):
                        if(prevMove == Directions.SOUTH and (not gameState.getWalls()[x-1][y] or not gameState.getWalls()[x+1][y])):
                            print("OBSTACULO ESTE/OESTE - BORDEANDO FIN")
                            move_random = random.randint(0, 1)
                            print("RAND", move_random)
                            if (move_random == 0):
                                if(Directions.WEST in legal):
                                    move = 0
                                elif(Directions.EAST in legal):
                                    move = 1
                            elif (move_random == 1):
                                if(Directions.EAST in legal):
                                    move = 1
                                elif(Directions.WEST in legal):
                                    move = 0
                        elif(prevMove == Directions.EAST and gameState.getWalls()[x][y+1] and Directions.EAST in legal):
                            move = 1
                            print("OBSTACULO NORTE - BORDEANDO DER")
                        elif(prevMove == Directions.WEST and gameState.getWalls()[x][y+1] and Directions.WEST in legal):
                            move = 0
                            print("OBSTACULO NORTE - BORDEANDO IZQ")
                        elif(Directions.NORTH not in legal):
                            print("OBSTACULO NORTE")
                            left = x
                            right = x

                            while left > 0 and gameState.getWalls()[left][y+1]:
                                left -= 1

                            while right < gameState.data.layout.width and gameState.getWalls()[right][y+1]:
                                right += 1

                            goLeft = x - left < right - x

                            if((x+right == gameState.data.layout.width or goLeft or Directions.EAST not in legal) and Directions.WEST in legal):
                                move = 0
                            elif((x-left == 0 or not goLeft or Directions.WEST not in legal) and Directions.EAST in legal):
                                move = 1
                            elif(Directions.SOUTH in legal):
                                move = 3
                        else:
                            move = 2
                            print("PREFERENCIA NORTE")
                    # PREFERENCIA SUR
                    else:
                        if(prevMove == Directions.NORTH and (not gameState.getWalls()[x-1][y] or not gameState.getWalls()[x+1][y])):
                            print("OBSTACULO ESTE/OESTE - BORDEANDO FIN")
                            move_random = random.randint(0, 1)
                            if (move_random == 0):
                                if(Directions.WEST in legal):
                                    move = 0
                                elif(Directions.EAST in legal):
                                    move = 1
                            elif (move_random == 1) and Directions.EAST in legal:
                                if(Directions.EAST in legal):
                                    move = 1
                                elif(Directions.WEST in legal):
                                    move = 0
                        elif(prevMove == Directions.EAST and gameState.getWalls()[x][y-1] and Directions.EAST in legal):
                            move = 1
                            print("OBSTACULO SUR - BORDEANDO DER")
                        elif(prevMove == Directions.WEST and gameState.getWalls()[x][y-1] and Directions.WEST in legal):
                            move = 0
                            print("OBSTACULO SUR - BORDEANDO IZQ")
                        elif(Directions.SOUTH not in legal):
                            print("OBSTACULO SUR")
                            left = x
                            right = x

                            while left > 0 and gameState.getWalls()[left][y-1]:
                                left -= 1
                            while right < gameState.data.layout.width and gameState.getWalls()[right][y-1]:
                                right += 1
                            goLeft = x - left < right - x

                            if((x+right == gameState.data.layout.width or goLeft or Directions.EAST not in legal) and Directions.WEST in legal):
                                move = 0
                            elif((x-left == 0 or not goLeft or Directions.WEST not in legal) and Directions.EAST in legal):
                                move = 1
                            elif(Directions.NORTH in legal):
                                move = 2
                        else:
                            move = 3
                            print("PREFERENCIA SUR")
            #print(">", move)
        except Exception:
            '''END GAME'''

        return move

    def getReward(self, previousGameState, gameState):
        try:
            previous_pacman_position = previousGameState.getPacmanPosition()
            previous_ghost_positions = previousGameState.getGhostPositions()
            previous_nearest_ghost_index = previousGameState.data.ghostDistances.index(min(
                value for value in previousGameState.data.ghostDistances if value is not None))

            pacman_position = gameState.getPacmanPosition()
            ghost_positions = gameState.getGhostPositions()
            nearest_ghost_index = gameState.data.ghostDistances.index(
                min(value for value in gameState.data.ghostDistances if value is not None))

            if(nearest_ghost_index == previous_nearest_ghost_index):

                previous_nearest_ghost = previous_ghost_positions[previous_nearest_ghost_index]
                nearest_ghost = ghost_positions[nearest_ghost_index]

                distlayout = distanceCalculator.computeDistances(
                    gameState.data.layout)
                prevDistance = distanceCalculator.getDistanceOnGrid(
                    distlayout, previous_pacman_position, previous_nearest_ghost)
                nextDistance = distanceCalculator.getDistanceOnGrid(
                    distlayout, pacman_position, nearest_ghost)
                reward = prevDistance - nextDistance
            elif(gameState.data.scoreChange > 0):
                reward = gameState.data.scoreChange
                print("TE COMI")
            else:
                reward = 0
        except Exception:
            reward = 1
        return reward

    def printStateData(self, state_data):
        if(state_data == 0):
            '''LEFT'''
            state_info = "LEFT"

        elif(state_data == 1):
            '''RIGHT'''
            state_info = "RIGHT"

        elif(state_data == 2):
            ''' UP '''
            state_info = "UP"
        else:
            '''DOWN'''
            state_info = "DOWN"

        return state_info

    def computePosition(self, state):
        """
        Compute the row of the qtable for a given state.
        For instance, the state (3,1) is the row 7
        """
        # print(state)   modified
        return state

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        position = self.computePosition(state)
        action_column = self.actions[action]
        return self.q_table[position][action_column]

    def computeValueFromQValues(self, gameState, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        legalActions = gameState.getLegalActions(0)
        if len(legalActions) == 0:
            return 0
        return max(self.q_table[self.computePosition(state)])

    def computeActionFromQValues(self, gameState, state):
        """
          Compute the best action to take in a state (If two or more states have same qvalue pick up randomly).  
          Note that if there are no legal actions, which is the case at the terminal state, you should return None.
        """
        legalActions = gameState.getLegalActions(0)
        if "Stop" in legalActions:
            legalActions.remove("Stop")
        if len(legalActions) == 0:
            return None
        best_actions = [legalActions[0]]
        best_value = self.getQValue(state, legalActions[0])
        for action in legalActions:
            value = self.getQValue(state, action)
            if value == best_value:
                best_actions.append(action)
            if value > best_value:
                best_actions = [action]
                best_value = value

        return random.choice(best_actions)

    def getAction(self, gameState):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
        """
        # Pick Action
        # legal = gameState.getLegalActions(0) ##Legal position from the pacman
        # legalActions = gameState..getLegalActions(state)

        state = self.getStateData(gameState)  # MODIFIED#
        legalActions = gameState.getLegalActions(0)
        if "Stop" in legalActions:
            legalActions.remove("Stop")
        action = None
        if len(legalActions) == 0:
            return action
        flip = util.flipCoin(self.epsilon)
        if flip:
            return random.choice(legalActions)
        return self.getPolicy(gameState, state)

    def update(self, gameState, current_state, action, next_state, reward):
        """
        The parent class calls this to observe a
        state = action => nextState and reward transition.
        You should do your Q-Value update here

        Good Terminal state -> reward 1
        Bad Terminal state -> reward -1
        Otherwise -> reward 0

        Q-Learning update:
            if terminal_state:
            Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + 0)
            else:
            Q(state,action) <- (1-self.alpha) Q(state,action) + self.alpha * (r + self.discount * max a' Q(nextState, a'))

        """
        # TRACE for transition and position to update. Comment the following lines if you do not want to see that trace

        position = self.computePosition(current_state)
        action_num = self.actions.get(action)
        legalActions = gameState.getLegalActions(0)

        if len(legalActions) == 1:
            self.q_table[position][action_num] = (1 - self.alpha) * self.q_table[position][action_num] + \
                self.alpha * (reward + 0)
        else:
            self.q_table[position][action_num] = (1 - self.alpha) * self.q_table[position][action_num] + \
                self.alpha * (reward + self.discount *
                              self.computeValueFromQValues(gameState, next_state))

        # TRACE for updated q-table. Comment the following lines if you do not want to see that trace
        # print("Q-table:")
        # self.printQtable()

    def getPolicy(self, gameState, state):
        "Return the best action in the qtable for a given state"
        return self.computeActionFromQValues(gameState, state)

    def getValue(self, state):
        "Return the highest q value for a given state"
        return self.computeValueFromQValues(state)

    def printQtable(self):
        "Print qtable"
        for line in self.q_table:
            print(line)
        print("\n")
