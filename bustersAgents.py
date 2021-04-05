from __future__ import print_function
import sys
import random
from distanceCalculator import Distancer
from game import Actions

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
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters

from wekaI import Weka

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

        self.weka = Weka()          #CUSTOM
        self.weka.start_jvm()       #CUSTOM


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

        x = []

        if "North" in  gameState.getLegalPacmanActions():
            x.append(True)
        else:
            x.append(False)

        if "South" in  gameState.getLegalPacmanActions():
            x.append(True)
        else:
            x.append(False)

        if "East" in  gameState.getLegalPacmanActions():
            x.append(True)
        else:
            x.append(False)

        if "West" in  gameState.getLegalPacmanActions():
            x.append(True)
        else:
            x.append(False)

        #x.append(True)

        # Pacman direction
        x.append(gameState.data.agentStates[0].getDirection())

        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        for livinGhost in gameState.getLivingGhosts()[1:]:
            x.append(livinGhost)

        # Ghosts positions
        for i in range(0, gameState.getNumAgents()-1):
            data = ','.join(map(str, gameState.getGhostPositions()[i]))
            x.append(data)

        # Ghosts directions
        data = ','.join(map(str, [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)]))
        x.append(data)

        # Manhattan distance to ghosts
        for ghostDistance in gameState.data.ghostDistances:
            if ghostDistance == None:
                x.append(-1)
            x.append(ghostDistance)
        
        # Manhattan distance to the closest pac dot
        if gameState.getDistanceNearestFood() == None:
            x.append(-1)
        else: 
            x.append(gameState.getDistanceNearestFood())

        #Last score
        x.append(gameState.data.score)
        print(x)

        a = self.weka.predict("./models/Perceptron1.model", x, "./!new_era/classification/keyboard/training_keyboard.arff")
        return a
   
    def printLineData(self, gameState, action, newGameState):

        # Pacman position
        data = ','.join(map(str, gameState.getPacmanPosition()))
        #msg = "Pacman position:"+data+","
        msg = data+","

        # Legal actions for Pacman in current position

        if "North" in  gameState.getLegalPacmanActions():
            data = "True,"
        else:
            data = "False,"

        if "South" in  gameState.getLegalPacmanActions():
            data += "True,"
        else:
            data += "False,"

        if "East" in  gameState.getLegalPacmanActions():
            data += "True,"
        else:
            data += "False,"

        if "West" in  gameState.getLegalPacmanActions():
            data += "True,"
        else:
            data += "False,"
        
        data+= "True,"
        msg+= data

        #msg += data+","

        # Pacman direction
        data = gameState.data.agentStates[0].getDirection()
        #msg += "Pacman direction: " + data 
        msg += data+","

        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        for livinGhost in gameState.getLivingGhosts()[1:]:

            msg += str(livinGhost)+","
        #data = ','.join(map(str, gameState.getLivingGhosts()))
        #msg += "Living ghosts: "+data+","
        #msg += data+","

        # Ghosts positions
        for i in range(0, gameState.getNumAgents()-1):
            data = ','.join(map(str, gameState.getGhostPositions()[i]))
            msg += data+"," 
        # Ghosts directions
        data = ','.join(map(str, [gameState.getGhostDirections().get(
            i) for i in range(0, gameState.getNumAgents() - 1)]))
        #msg += "Ghosts directions: "+data+","
        msg += data+","
        # Manhattan distance to ghosts
        for ghostDistance in gameState.data.ghostDistances:
            if ghostDistance == None:
                ghostDistance = -1
            msg += str(ghostDistance)+","
        
        #msg += "Ghosts distances: "+data+","
        #msg += data+","
        # Manhattan distance to the closest pac dot
        if gameState.getDistanceNearestFood() == None:
            msg += str(-1)+","
        else: msg += str(gameState.getDistanceNearestFood())+","
        #msg += "Distance nearest pac dots: " + data
        
        #Last score
        msg+=str(gameState.data.score)+","

        #Next scoreChange
        msg+=str(newGameState.data.scoreChange)+","

        #Last action
        msg+= str(action)+","
        
        #Next score
        msg+=str(newGameState.data.score)+"\n"
        return msg  

class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index=0, inference="KeyboardInference", ghostAgents=None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        return KeyboardAgent.getAction(self, gameState)

    def printLineData(self, gameState, action, newGameState):

        # Pacman position
        data = ','.join(map(str, gameState.getPacmanPosition()))
        #msg = "Pacman position:"+data+","
        msg = data+","

        # Legal actions for Pacman in current position

        if "North" in  gameState.getLegalPacmanActions():
            data = "True,"
        else:
            data = "False,"

        if "South" in  gameState.getLegalPacmanActions():
            data += "True,"
        else:
            data += "False,"

        if "East" in  gameState.getLegalPacmanActions():
            data += "True,"
        else:
            data += "False,"

        if "West" in  gameState.getLegalPacmanActions():
            data += "True,"
        else:
            data += "False,"
        
        data+= "True,"
        msg+= data

        #msg += data+","

        # Pacman direction
        data = gameState.data.agentStates[0].getDirection()
        #msg += "Pacman direction: " + data 
        msg += data+","

        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        for livinGhost in gameState.getLivingGhosts()[1:]:

            msg += str(livinGhost)+","
        #data = ','.join(map(str, gameState.getLivingGhosts()))
        #msg += "Living ghosts: "+data+","
        #msg += data+","

        # Ghosts positions
        for i in range(0, gameState.getNumAgents()-1):
            data = ','.join(map(str, gameState.getGhostPositions()[i]))
            msg += data+"," 
        # Ghosts directions
        data = ','.join(map(str, [gameState.getGhostDirections().get(
            i) for i in range(0, gameState.getNumAgents() - 1)]))
        #msg += "Ghosts directions: "+data+","
        msg += data+","
        # Manhattan distance to ghosts
        for ghostDistance in gameState.data.ghostDistances:
            if ghostDistance == None:
                ghostDistance = -1
            msg += str(ghostDistance)+","
        
        #msg += "Ghosts distances: "+data+","
        #msg += data+","
        # Manhattan distance to the closest pac dot
        if gameState.getDistanceNearestFood() == None:
            msg += str(-1)+","
        else: msg += str(gameState.getDistanceNearestFood())+","
        #msg += "Distance nearest pac dots: " + data
        
        #Last score
        msg+=str(gameState.data.score)+","

        #Next scoreChange
        msg+=str(newGameState.data.scoreChange)+","

        #Last action
        msg+= str(action)+","
        
        #Next score
        msg+=str(newGameState.data.score)+"\n"
        return msg  

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
    
    def printLineData(self, gameState, action, newGameState):

        # Pacman position
        data = ','.join(map(str, gameState.getPacmanPosition()))
        #msg = "Pacman position:"+data+","
        msg = data+","

        # Legal actions for Pacman in current position

        if "North" in  gameState.getLegalPacmanActions():
            data = "True,"
        else:
            data = "False,"

        if "South" in  gameState.getLegalPacmanActions():
            data += "True,"
        else:
            data += "False,"

        if "East" in  gameState.getLegalPacmanActions():
            data += "True,"
        else:
            data += "False,"

        if "West" in  gameState.getLegalPacmanActions():
            data += "True,"
        else:
            data += "False,"
        
        data+= "True,"
        msg+= data

        #msg += data+","

        # Pacman direction
        data = gameState.data.agentStates[0].getDirection()
        #msg += "Pacman direction: " + data 
        msg += data+","

        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        for livinGhost in gameState.getLivingGhosts()[1:]:

            msg += str(livinGhost)+","
        #data = ','.join(map(str, gameState.getLivingGhosts()))
        #msg += "Living ghosts: "+data+","
        #msg += data+","

        # Ghosts positions
        for i in range(0, gameState.getNumAgents()-1):
            data = ','.join(map(str, gameState.getGhostPositions()[i]))
            msg += data+"," 
        # Ghosts directions
        data = ','.join(map(str, [gameState.getGhostDirections().get(
            i) for i in range(0, gameState.getNumAgents() - 1)]))
        #msg += "Ghosts directions: "+data+","
        msg += data+","
        # Manhattan distance to ghosts
        for ghostDistance in gameState.data.ghostDistances:
            if ghostDistance == None:
                ghostDistance = -1
            msg += str(ghostDistance)+","
        
        #msg += "Ghosts distances: "+data+","
        #msg += data+","
        # Manhattan distance to the closest pac dot
        if gameState.getDistanceNearestFood() == None:
            msg += str(-1)+","
        else: msg += str(gameState.getDistanceNearestFood())+","
        #msg += "Distance nearest pac dots: " + data
        
        #Last score
        msg+=str(gameState.data.score)+","

        #Next scoreChange
        msg+=str(newGameState.data.scoreChange)+","

        #Last action
        msg+= str(action)+","
        
        #Next score
        msg+=str(newGameState.data.score)+"\n"
        return msg  

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
    
    def printLineData(self, gameState, action, newGameState):

        # Pacman position
        data = ','.join(map(str, gameState.getPacmanPosition()))
        #msg = "Pacman position:"+data+","
        msg = data+","

        # Legal actions for Pacman in current position

        if "North" in  gameState.getLegalPacmanActions():
            data = "True,"
        else:
            data = "False,"

        if "South" in  gameState.getLegalPacmanActions():
            data += "True,"
        else:
            data += "False,"

        if "East" in  gameState.getLegalPacmanActions():
            data += "True,"
        else:
            data += "False,"

        if "West" in  gameState.getLegalPacmanActions():
            data += "True,"
        else:
            data += "False,"
        
        data+= "True,"
        msg+= data

        #msg += data+","

        # Pacman direction
        data = gameState.data.agentStates[0].getDirection()
        #msg += "Pacman direction: " + data 
        msg += data+","

        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        for livinGhost in gameState.getLivingGhosts()[1:]:

            msg += str(livinGhost)+","
        #data = ','.join(map(str, gameState.getLivingGhosts()))
        #msg += "Living ghosts: "+data+","
        #msg += data+","

        # Ghosts positions
        for i in range(0, gameState.getNumAgents()-1):
            data = ','.join(map(str, gameState.getGhostPositions()[i]))
            msg += data+"," 
        # Ghosts directions
        data = ','.join(map(str, [gameState.getGhostDirections().get(
            i) for i in range(0, gameState.getNumAgents() - 1)]))
        #msg += "Ghosts directions: "+data+","
        msg += data+","
        # Manhattan distance to ghosts
        for ghostDistance in gameState.data.ghostDistances:
            if ghostDistance == None:
                ghostDistance = -1
            msg += str(ghostDistance)+","
        
        #msg += "Ghosts distances: "+data+","
        #msg += data+","
        # Manhattan distance to the closest pac dot
        if gameState.getDistanceNearestFood() == None:
            msg += str(-1)+","
        else: msg += str(gameState.getDistanceNearestFood())+","
        #msg += "Distance nearest pac dots: " + data
        
        #Last score
        msg+=str(gameState.data.score)+","

        #Next scoreChange
        msg+=str(newGameState.data.scoreChange)+","

        #Last action
        msg+= str(action)+","
        
        #Next score
        msg+=str(newGameState.data.score)+"\n"
        return msg  

class BasicAgentAA(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0

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
        # self.printInfo(gameState)

        move = Directions.STOP
        legal = gameState.getLegalActions(0)  # Legal position from the pacman

        # Manhattan distance to ghosts
        ghostsDistances = gameState.data.ghostDistances

        minGhostsDistance = min(x for x in ghostsDistances if x is not None)

        '''if(gameState.getDistanceNearestFood() is not None and minGhostsDistance > gameState.getDistanceNearestFood()):  # Move twoards nearst dot
        print("Pacdots")
            move_random = random.randint(0, 3)
            if (move_random == 0) and Directions.WEST in legal:
                move = Directions.WEST
            if (move_random == 1) and Directions.EAST in legal:
                move = Directions.EAST
            if (move_random == 2) and Directions.NORTH in legal:
                move = Directions.NORTH
            if (move_random == 3) and Directions.SOUTH in legal:
                move = Directions.SOUTH
            

        else:  # Move twoards Ghost'''

        ghostPosition = gameState.getGhostPositions()[ghostsDistances.index(minGhostsDistance)]
        validMovements = gameState.getLegalPacmanActions()
        pacPosition = gameState.getPacmanPosition()
        xDistanceToGhost = abs(ghostPosition[0] - pacPosition[0])
        yDistanceToGhost = abs(ghostPosition[1] - pacPosition[1])
        prevMove = gameState.data.agentStates[0].getDirection()
        x, y = gameState.getPacmanPosition()
        print("$",prevMove)
        if(len(validMovements) == 5 and not gameState.getWalls()[x+1][y+1] and not gameState.getWalls()[x+1][y-1] and not gameState.getWalls()[x-1][y+1] and not gameState.getWalls()[x-1][y-1]):
            print("LIBRE")
        
            if(xDistanceToGhost>yDistanceToGhost):

                if(ghostPosition[0] > pacPosition[0]):
                    move = Directions.EAST
                elif(ghostPosition[0] < pacPosition[0]):
                    move = Directions.WEST
            else:
                if(ghostPosition[1] > pacPosition[1]):
                    move = Directions.NORTH
                else:
                    move = Directions.SOUTH
        else:
            #EJE X
            if(xDistanceToGhost>yDistanceToGhost):
                #PREFERENCIA ESTE
                if(ghostPosition[0] > pacPosition[0]):
                    '''if((prevMove==Directions.NORTH or prevMove==Directions.SOUTH) and not gameState.getWalls()[x+1][y] and Directions.EAST in legal): 
                        move = Directions.EAST
                        print("OBSTACULO ESTE - BORDEANDO FIN")'''
                    
                    if(prevMove==Directions.WEST and (not gameState.getWalls()[x][y+1] or not gameState.getWalls()[x][y-1])):
                        print("OBSTACULO NORTE/SUR - BORDEANDO FIN")
                        move_random = random.randint(0, 1)
                        if (move_random == 0) and Directions.NORTH in legal:
                            move = Directions.NORTH
                        elif (move_random == 1) and Directions.SOUTH in legal:
                            move = Directions.SOUTH          

                    elif(prevMove==Directions.NORTH and gameState.getWalls()[x+1][y]and Directions.NORTH in legal): 
                        move = Directions.NORTH
                        print("OBSTACULO ESTE - BORDEANDO ARRIBA")
                    
                    elif(prevMove==Directions.SOUTH and gameState.getWalls()[x+1][y] and Directions.SOUTH in legal): 
                        move = Directions.SOUTH
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

                        print("y    ",y)
                        print("up    ",up)
                        print("down    ",down)

                        if((y-down == 0 or goUp or Directions.SOUTH not in legal) and Directions.NORTH in legal):
                            move = Directions.NORTH
                        elif((up+y == gameState.data.layout.height or not goUp or Directions.NORTH not in legal)and Directions.SOUTH in legal):
                            move = Directions.SOUTH
                        elif(Directions.WEST in legal):
                             move = Directions.WEST
                    else:
                        print("PREFERENCIA ESTE")
                        move = Directions.EAST
                        

                #PREFERENCIA OSTE
                else:
                    '''if((prevMove==Directions.NORTH or prevMove==Directions.SOUTH) and not gameState.getWalls()[x-1][y] and Directions.WEST in legal): 
                        move = Directions.WEST
                        print("OBSTACULO OESTE - BORDEANDO FIN")'''
                                   
                    if(prevMove==Directions.EAST and (not gameState.getWalls()[x][y+1] or not gameState.getWalls()[x][y-1])):
                        print("OBSTACULO NORTE/SUR - BORDEANDO FIN")
                        move_random = random.randint(0, 1)
                        if (move_random == 0) and Directions.NORTH in legal:
                            move = Directions.NORTH
                        elif (move_random == 1) and Directions.SOUTH in legal:
                            move = Directions.SOUTH          
                    elif(prevMove==Directions.NORTH and gameState.getWalls()[x-1][y] and Directions.NORTH) in legal: 
                        move = Directions.NORTH
                        print("OBSTACULO OESTE - BORDEANDO ARRIBA")
                    
                    elif(prevMove==Directions.SOUTH and gameState.getWalls()[x-1][y] and Directions.SOUTH in legal): 
                        move = Directions.SOUTH
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
                        if((y-down==0 or goUp or Directions.SOUTH not in legal) and Directions.NORTH in legal):
                            move = Directions.NORTH
                        elif((up+y == gameState.data.layout.height or not goUp or Directions.NORTH not in legal) and Directions.SOUTH in legal):
                            move = Directions.SOUTH
                        elif(Directions.EAST in legal):
                             move = Directions.EAST

                    else:
                        move = Directions.WEST
                        print("PREFERENCIA OESTE")
            
            #EJE Y    
            else:
                #PREFERENCIA NORTE
                if(ghostPosition[1] > pacPosition[1]):

                    '''if((prevMove==Directions.EAST or prevMove==Directions.WEST) and not gameState.getWalls()[x][y+1] and Directions.NORTH in legal): 
                        move = Directions.NORTH
                        print("OBSTACULO NORTE - BORDEANDO FIN")'''
                                                    
                    if(prevMove==Directions.SOUTH and (not gameState.getWalls()[x-1][y] or not gameState.getWalls()[x+1][y])):
                        print("OBSTACULO ESTE/OESTE - BORDEANDO FIN")
                        move_random = random.randint(0, 1)
                        if (move_random == 0) and Directions.WEST in legal:
                            move = Directions.WEST
                        elif (move_random == 1) and Directions.EAST in legal:
                            move = Directions.EAST          

                    elif(prevMove==Directions.EAST and gameState.getWalls()[x][y+1] and Directions.EAST in legal): 
                        move = Directions.EAST
                        print("OBSTACULO NORTE - BORDEANDO DER")
                    
                    elif(prevMove==Directions.WEST and gameState.getWalls()[x][y+1] and Directions.WEST in legal): 
                        move = Directions.WEST
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

                        if((x+right==gameState.data.layout.width or goLeft or Directions.EAST not in legal) and Directions.WEST in legal):
                            move = Directions.WEST
                        elif((x-left==0 or not goLeft or Directions.WEST not in legal) and Directions.EAST in legal):
                            move = Directions.EAST
                        elif(Directions.SOUTH in legal):
                             move = Directions.SOUTH
                    else:
                        move = Directions.NORTH
                        print("PREFERENCIA NORTE")
                            
                #PREFERENCIA SUR
                else:

                    '''if((prevMove==Directions.EAST or prevMove==Directions.WEST) and not gameState.getWalls()[x][y-1] and Directions.SOUTH in legal): 
                    move = Directions.SOUTH
                    print("OBSTACULO SUR - BORDEANDO FIN")'''

                    if(prevMove==Directions.NORTH and (not gameState.getWalls()[x-1][y] or not gameState.getWalls()[x+1][y])):
                        print("OBSTACULO ESTE/OESTE - BORDEANDO FIN")
                        move_random = random.randint(0, 1)
                        if (move_random == 0) and Directions.WEST in legal:
                            move = Directions.WEST
                        elif (move_random == 1) and Directions.EAST in legal:
                            move = Directions.EAST                     
                        
                    elif(prevMove==Directions.EAST and gameState.getWalls()[x][y-1] and Directions.EAST in legal): 
                        move = Directions.EAST
                        print("OBSTACULO SUR - BORDEANDO DER")
                    elif(prevMove==Directions.WEST and gameState.getWalls()[x][y-1] and Directions.WEST in legal): 
                        move = Directions.WEST
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

                        if((x+right==gameState.data.layout.width or goLeft or Directions.EAST not in legal) and Directions.WEST in legal):
                            move = Directions.WEST
                        elif((x-left==0 or not goLeft or Directions.WEST not in legal) and Directions.EAST in legal):
                            move = Directions.EAST
                        elif(Directions.NORTH in legal):
                             move = Directions.NORTH
                    else:
                        move = Directions.SOUTH
                        print("PREFERENCIA SUR")
        
        print(">",move)
    
        return move

    def printLineData(self, gameState, action, newGameState):

        # Pacman position
        data = ','.join(map(str, gameState.getPacmanPosition()))
        #msg = "Pacman position:"+data+","
        msg = data+","

        # Legal actions for Pacman in current position

        if "North" in  gameState.getLegalPacmanActions():
            data = "True,"
        else:
            data = "False,"

        if "South" in  gameState.getLegalPacmanActions():
            data += "True,"
        else:
            data += "False,"

        if "East" in  gameState.getLegalPacmanActions():
            data += "True,"
        else:
            data += "False,"

        if "West" in  gameState.getLegalPacmanActions():
            data += "True,"
        else:
            data += "False,"
        
        data+= "True,"
        msg+= data

        #msg += data+","

        # Pacman direction
        data = gameState.data.agentStates[0].getDirection()
        #msg += "Pacman direction: " + data 
        msg += data+","

        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        for livinGhost in gameState.getLivingGhosts()[1:]:

            msg += str(livinGhost)+","
        #data = ','.join(map(str, gameState.getLivingGhosts()))
        #msg += "Living ghosts: "+data+","
        #msg += data+","

        # Ghosts positions
        for i in range(0, gameState.getNumAgents()-1):
            data = ','.join(map(str, gameState.getGhostPositions()[i]))
            msg += data+"," 
        # Ghosts directions
        data = ','.join(map(str, [gameState.getGhostDirections().get(
            i) for i in range(0, gameState.getNumAgents() - 1)]))
        #msg += "Ghosts directions: "+data+","
        msg += data+","
        # Manhattan distance to ghosts
        for ghostDistance in gameState.data.ghostDistances:
            if ghostDistance == None:
                ghostDistance = -1
            msg += str(ghostDistance)+","
        
        #msg += "Ghosts distances: "+data+","
        #msg += data+","
        # Manhattan distance to the closest pac dot
        if gameState.getDistanceNearestFood() == None:
            msg += str(-1)+","
        else: msg += str(gameState.getDistanceNearestFood())+","
        #msg += "Distance nearest pac dots: " + data
        
        #Last score
        msg+=str(gameState.data.score)+","

        #Next scoreChange
        msg+=str(newGameState.data.scoreChange)+","

        #Last action
        msg+= str(action)+","
        
        #Next score
        msg+=str(newGameState.data.score)+"\n"
        return msg  