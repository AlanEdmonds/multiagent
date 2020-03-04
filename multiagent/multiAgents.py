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
import math

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        #returns a best action, as determined by applying evaluationFunction on all legal actions
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
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates() #the list of ghost agentstates, each has functions such as getPosition and getDirection
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        #brainstorming: implementations taking different factors into account, in order of increasing difficulty:
            #1) collects nearest food
            #2) weight a move based on a sum of the number of foods it brings closer to pacman with exponentially heavier weights depending on
            #       the proximity to pacman, minus [something equivalent for the foods that become farther away] * [some constant c<1]
            #3) take ghost positions into account: weight each ghost with exponentially heavier weights depending on proximity to pacman (ideally with
            #       a higher exponent than is used for foods). negative value if action takes pacman closer to ghost, positive if takes away from ghost.
        ###########################
        #mazeDistance calc stuff
        ###########################
        def manhattanDistance(point1, point2):
            x1, x2 = point1
            y1, y2 = point2
            return (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)
        def mazeDistance(point1, point2):
            return len(bfs(point1, point2))
        def bfs(start, goal):
            from util import Queue
            q = Queue()
            explored = set()
            q.push((start, []))
            while not q.isEmpty():
                popped = q.pop()
                if popped[0] in explored:
                    continue
                explored.add(popped[0])
                if popped[0] == goal:
                    return popped[1]
                for successor in getSuccessors(popped[0]):
                    if successor[0] not in explored:
                        #push successor, where actions and cost are cumulative on its parent (popped)
                        q.push((successor[0], popped[1] + [successor[1]]))
            return popped[1]

        def getSuccessors(position):
            from game import Actions
            successors = []
            for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                x,y = position
                dx, dy = Actions.directionToVector(action)
                nextx, nexty = int(x + dx), int(y + dy)
                walls = successorGameState.getWalls()
                if not walls[nextx][nexty]:
                    nextState = (nextx, nexty)
                    successors.append((nextState, action))
            return successors
        ###########################
        #mazeDistance calc stuff
        ###########################



        ###########################
        #food stuff
        ###########################
        oldPos = currentGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        w = oldFood.width
        h = oldFood.height
        mazeSize = math.sqrt(w*w + h*h)
        closerF = [] #list of distances from newPos to food for the foods that become closer after action. each distance is in a tuple also containing the foods coords
        fartherF = [] #become farther
        for i in range(w):
            for j in range(h):
                if oldFood[i][j]: #theres a food at (i,j) in oldfood, implying theres a food in newfood, apart from in the following edge case
                    if not newFood[i][j]: #this would mean we just ate the food at (i,j)
                        #print("(should be true) we ate food! ", newPos == (i, j))
                        closerF.append((.7, (i, j))) #for the sake of calculations we equate eating a food to having a food only .7 distance away from pacman
                        continue
                    oldDist = mazeDistance(oldPos, (i, j))
                    newDist = mazeDistance(newPos, (i, j))
                    if oldDist > newDist:
                        closerF.append((newDist, (i, j)))
                    if oldDist < newDist:
                        fartherF.append((newDist, (i, j)))
        ###########################
        #food stuff
        ###########################



        ###########################
        #ghost stuff
        ###########################
        closerG = [] #list of distances to ghosts that become closer
        fartherG = [] #become farther
        ghostPositions = []
        for i in range(len(newGhostStates)):
            x,y = newGhostStates[i].getPosition()
            ghostPositions.append((int(x), int(y)))
        for i in range(len(newGhostStates)):
            if newPos == ghostPositions[i]: #we run into a ghost
                closerG.append((0.001, newGhostStates[i]))
                continue
            oldDist = mazeDistance(oldPos, ghostPositions[i])
            newDist = mazeDistance(newPos, ghostPositions[i])
            if oldDist > newDist:
                closerG.append((newDist, newGhostStates[i]))
            if oldDist < newDist:
                fartherG.append((newDist, newGhostStates[i]))
        ###########################
        #ghost stuff
        ###########################



        ###########################
        #corner stuff
        ###########################
        corners = [(w,1), (w,h), (1,h), (1,1)]
        x,y = int(newPos[0]), int(newPos[1])
        closest = math.sqrt(manhattanDistance((1,1), (x,y))) #dist from (1,1)
        index = 3 #len(corners) - 1
        for i in range(len(corners) - 1): #closest already set to dist from (1,1), so we only iterate over len - 1
            c = corners[i]
            manhattanD = math.sqrt(manhattanDistance(c, (x,y)))
            if manhattanD < closest:
                closest = manhattanD
                index = i
        pnc = corners[index]    #there was a shorter way to do this based on looking at newPos coords vs w/2 and h/2....

        def pncManhattanDist(tuple):
            d = math.sqrt(manhattanDistance(tuple, pnc)) #manhattan distance to pnc
            if d > mazeSize/2: #if d is greater than half the grids hypotenuse
                return -1
            #else, it is in the same quadrant as pacman. we'll value the food more for smaller d
            if d == 0:
                return 0.5
            return d
        ###########################
        #corner stuff
        ###########################



        ###########################
        #score calc stuff
        ###########################
        wallsNum = 0
        wallGrid = successorGameState.getWalls()
        for i in range(wallGrid.width): #apparently the same value as w
            for j in range(wallGrid.height):
                if wallGrid[i][j]:
                    wallsNum += 1
        initialFood = wallGrid.width*wallGrid.height - wallsNum - (wallGrid.height-2)
        foodLeft = currentGameState.getNumFood()
        foodleftMultiplier = min(1, initialFood/foodLeft/4)
        closerFoodScore = 0
        fartherFoodScore = 0
        if foodLeft == 0:
            foodLeft = 0.5
        for x in closerF:
            dist, coords = x[0], x[1]
            n = pncManhattanDist(coords)
            cornerMultiplier = 1
            if n > 0:
                cornerMultiplier = min(1, mazeSize/2/n / 4)
            val = 1/dist/dist*foodleftMultiplier*cornerMultiplier
            closerFoodScore += val

        """"
        for x in fartherF:
            dist, coords = x[0], x[1]
            n = pncManhattanDist(coords)
            cornerMultiplier = 1
            if n > 0:
                cornerMultiplier = min(1, mazeSize/2/n / 4)
            val = 1/dist/dist*foodleftMultiplier*cornerMultiplier
            fartherFoodScore += val
        """
        ghostScore = 0
        nearestScared = None
        for ghost in closerG: #find nearest scared ghost
            if ghost[1].scaredTimer > 0:
                if not nearestScared:
                    nearestScared = ghost
                if ghost[0] < nearestScared[0]:
                    nearestScared = ghost

        for ghost in closerG: #tuple of dist to newPos and the ghost object
            val = 1/ghost[0]**(3)
            if ghost[1].scaredTimer == 0:
                if not nearestScared:
                    ghostScore -= val
                    continue
                if ghost[0] < nearestScared[0]:
                    print("asdf")
                    ghostScore -= val**(1/3)*nearestScared[0]/ghost[0] #if ghost is closer than nearestScared, lets make moving towards ghost much more costly. more costly if neared scared if far, more costly if brave ghost is close

            if ghost[1].scaredTimer > 0:
                if ghost[0] == 1:
                    ghostScore += math.sqrt(val)*(ghost[1].scaredTimer + 40)**3 #go in for the kill
                difference = ghost[1].scaredTimer - ghost[0]
                if difference >= 0:
                    ghostScore += math.sqrt(val)*(ghost[1].scaredTimer + 40)*(difference+1)**2
                ghostScore += math.sqrt(val)*ghost[1].scaredTimer

        for ghost in fartherG:
            val = 1/ghost[0]**(3)
            ghostScore += val

        #stop will always return 0 since oldDist == newDist always, so closer and farther lists remain empty
        return ghostScore + closerFoodScore #we no longer subtract fartherFoodScore/2, it doesnt seem beneficial
        """
        capsules = currentGameState.getCapsules()
        closeCapDist = mazeDistance(newPos, capsules[len(capsules) - 1]) #initialize with last capsule in list
        index = len(capsules) - 1
        oneAway = False
        for i in range(len(capsules) - 1):
            dist = mazeDistance(newPos, capsules[i])
            if dist < closeCapDist:
                closeCapDist = dist
                index = i
        if closeCapDist == 1:
            oneAway = True
        closestCap = capsules[index]


        #calculate (mazeDistance/2)**2/distance**4 for each ghost to closestcap, sum it, and mult by number of ghosts (number of ghosts is kindof to the power of 2-- its like we take the average distance for all ghosts and mult by num of ghosts squared)
        #divide by ourdistance to closestcap
        capsuleScore = 0
        for ghost in ghostPositions:
            capsuleScore += 1/mazeDistance(ghost, closestCap)**3
        numOfGhosts = len(ghostPositions)
        capsuleScore *= mazeSize/2numOfGhosts/mazeDistance(newPos, closestCap) #the mazeSize/2 at the front was chosen pretty arbitrarily, just to make capsulescore a bit bigger
        #if the action makes us eat a food and brings us to newPos thats one away from capsule, we make capsulescore super important
            #capsuleScore *= (mazeSize/2)**
            #if oneaway capsuleScore is some large negative value if we go closer, unless our current capsulescore is really good, then we pump it up super high
        oldDist = mazeDistance(oldPos, closestCap)
        newDist = mazeDistance(newPos, closestCap)
        #if oneAway
        #if oldDist == newDist:
        #    capsuleScore = 0
        #if oldDist < newDist:
        #    capsuleScore = -1*capsuleScore/numOfGhosts

        print(ghostScore, closerFoodScore, capsuleScore)
        """

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
        "*** YOUR CODE HERE"
        """
        def maxVal(state, agentCount, depth):
            if depth == 0 or state.isLose() or state.isWin():
                return self.evaluationFunction(state)
            value = -69000
            legalMoves = state.getLegalActions()
            for action in legalMoves:
                value = max(value, minVal(state.generateSuccessor(0, action), 1, agentCount, depth-1))
            return value

        def minVal(state, ghostIndex, agentCount, depth):
            value = 6969
            if depth == 0 or state.isLose() or state.isWin():
                return self.evaluationFunction(state)
            legalMoves = state.getLegalActions(ghostIndex)
            if ghostIndex == agentCount - 1: #we've come to the last ghost-- its time to decrease depth and go back to moving pacman
                for action in legalMoves:
                    value = min(value, maxVal(state.generateSuccessor(ghostIndex, action), agentCount, depth-1))
                return value
            for action in legalMoves:
                value = min(value, minVal(state.generateSuccessor(ghostIndex, action), ghostIndex+1, agentCount, depth))
            return value

        legalMoves = gameState.getLegalActions()
        agentCount = gameState.getNumAgents()
        previous = -699999
        best = None
        for action in legalMoves:
            score = minVal(gameState.generateSuccessor(0, action), 1, agentCount, self.depth)
            if score > previous:
                best = action;
            previous = score
        return best
        """
        #above method not expanding right number of nodes... another attempt:
        def minimax(state, agentIndex, action, depth):
            agentCount = state.getNumAgents()
            if state.isLose() or state.isWin() or depth == 0:
                return (action, self.evaluationFunction(state))
            if agentIndex == 0:
                bestMove = (None, -69)
                for legalAction in state.getLegalActions(agentIndex):
                    if depth == self.depth*agentCount:
                        next = legalAction
                    else:
                        next = action
                    currentMove = minimax(state.generateSuccessor(agentIndex, legalAction), (agentIndex + 1) % agentCount, next, depth - 1)
                    if currentMove[1] > bestMove[1]:
                        bestMove = currentMove
                return bestMove
            else:
                bestAction = (None, 69000)
                for legalAction in state.getLegalActions(agentIndex):
                    currentMove = minimax(state.generateSuccessor(agentIndex, legalAction), (agentIndex + 1) % agentCount, action, depth - 1)
                    if currentMove[1] < bestAction[1]:
                        bestAction = currentMove
            return bestAction
        return minimax(gameState, 0, Directions.STOP, gameState.getNumAgents()*self.depth)[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        """
        def alphaBeta(gameState, action, depth, agentIndex, alpha, beta):
            if gameState.isLose() or gameState.isWin() or depth == 0:
                return (action, self.evaluationFunction(gameState))
            if agentIndex == 0:
                bestAction = (action, alpha)
                for legalAction in gameState.getLegalActions(agentIndex):
                    currAction = alphaBeta(gameState.generateSuccessor(agentIndex, legalAction), (legalAction if depth is self.depth*gameState.getNumAgents() else action), depth - 1, 1, alpha, beta)
                    if currAction[1] > bestAction[1]:
                        bestAction = currAction
                    if bestAction[1] > alpha:
                        break
                return bestAction
            else:
                bestAction = (Directions.STOP, beta)
                for legalAction in gameState.getLegalActions(agentIndex):
                    currAction = alphaBeta(gameState.generateSuccessor(agentIndex, legalAction), action, depth - 1, (agentIndex + 1) % gameState.getNumAgents(), alpha, beta)
                    if currAction[1] < bestAction[1]:
                        bestAction = currAction
                    if bestAction[1] < beta:
                        break
                return bestAction
        return alphaBeta(gameState, Directions.STOP, self.depth*gameState.getNumAgents(), 0, -999999, 999999)[0]
        """
        evaluationFunc = self.evaluationFunction

        # numOfAgents = gameState.getNumAgents()
        # numOfGhosts = numOfAgents-1

        alfa = -99999999
        beta = 99999999

        def maxValue(state, depth, alfa, beta):
            v = -9999999999
            #legalMoves = state.getLegalActions(0)
            # need to modify the for loop compared the previous question, because that one expands always.
            for act in state.getLegalActions(0):
                #successor = state.generateSuccessor(0, act)
                numOfAgents = state.getNumAgents()
                numOfGhosts = numOfAgents - 1

                if numOfGhosts == 0:
                    v = max(v, value(0, state.generateSuccessor(0, act), depth + 1, alfa, beta))
                    if v > beta:
                        #print '...............pruningg this max: ', v
                        return v

                    alfa = max(v, alfa)

                else:
                    v = max(v, value(1, state.generateSuccessor(0, act), depth, alfa, beta))
                    if v > beta:
                        #print '...............pruningg this max: ', v
                        return v
                    alfa = max(alfa, v)
            return v

        def minValue(state, depth, agentIndex, alfa, beta):
            v = 9999999999
            numOfAgents = state.getNumAgents()
            numOfGhosts = numOfAgents - 1

            for act in state.getLegalActions(agentIndex):
                #successor = state.generateSuccessor(agentIndex, act)
                numOfAgents = state.getNumAgents()
                numOfGhosts = numOfAgents - 1
                if agentIndex == numOfGhosts:
                    v = min(v, value(0, state.generateSuccessor(agentIndex, act), depth + 1, alfa, beta))
                    if v < alfa:
                        #print '...............pruningg this min: ', v
                        return v
                    beta = min(beta,v)
                    # print 'v after min operation: ', v
                else:
                    v = min(v, value(agentIndex + 1, state.generateSuccessor(agentIndex, act), depth, alfa, beta))
                    if v < alfa:
                        #print '...............pruningg this min: ', v
                        return v
                    beta = min(beta,v)
                    # print 'v after min operation: ', v

            return v

        def value(agentIndex, state, depth, alfa ,beta):
            if depth == self.depth:
                scr = evaluationFunc(state)
                #print 'evaluated value: ',scr
                return scr
            if state.isWin() or state.isLose():
                scr = evaluationFunc(state)
                #print 'evaluated value: ', scr
                return scr
            numOfAgents = state.getNumAgents()
            numOfGhosts = numOfAgents - 1
            if agentIndex == 0 or numOfGhosts == 0:
                return maxValue(state, depth, alfa, beta)

            return minValue(state, depth, agentIndex, alfa, beta)

        legalMoves = gameState.getLegalActions()

        numOfAgents = gameState.getNumAgents()
        numOfGhosts = numOfAgents - 1

        scores = []

        for act in legalMoves:
            successor = gameState.generateSuccessor(0, act)
            if numOfGhosts == 0:
                v = value(0, gameState.generateSuccessor(0, act), 1, alfa, beta)
            else:
                v = value(1, gameState.generateSuccessor(0, act), 0, alfa, beta)
            alfa = max(v, alfa)
            scores.append(v)

        bestScore = max(scores)
        # print 'best scores: ',bestScore
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = bestIndices[0]  # Pick first of the bests

        return legalMoves[chosenIndex]

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
        "*** YOUR CODE HERE ***"
        currentD = 0
        currentAI = 0
        val = self.value(gameState, currentD, currentAI)
        return val[0]

    def value(self, state, currentD, currentAI):
        if currentAI >= state.getNumAgents():
            currentAI = 0
            currentD += 1
        if currentD == self.depth:
            return self.evaluationFunction(state)
        if currentAI == 0:
            return self.maxValue(state, currentD, currentAI)
        else:
            return self.expValue(state, currentD, currentAI)

    def expValue(self, state, currentD, currentAI):
        v = ["unknown", 0]
        if not state.getLegalActions(currentAI):
            return self.evaluationFunction(state)

        prob = 1.0/len(state.getLegalActions(currentAI))
        for action in state.getLegalActions(currentAI):
            if action == "Stop":
                continue
            retVal = self.value(state.generateSuccessor(currentAI, action), currentD, currentAI + 1)
            if type(retVal) is tuple:
                retVal = retVal[1]
            v[1] += retVal * prob
            v[0] = action
        return tuple(v)

    def maxValue(self, state, currentD, currentAI):
        v = ("unknown", -1*float("inf"))
        if not state.getLegalActions(currentAI):
            return self.evaluationFunction(state)
        for action in state.getLegalActions(currentAI):
            if action == "Stop":
                continue
            retVal = self.value(state.generateSuccessor(currentAI, action), currentD, currentAI + 1)
            if type(retVal) is tuple:
                retVal = retVal[1]
            vNew = max(v[1], retVal)
            if vNew is not v[1]:
                v = (action, vNew)
        return v

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    """
    "*** YOUR CODE HERE ***"
    """
    def mazeDistance(point1, point2):
        return len(bfs(point1, point2))
    def bfs(start, goal):
        from util import Queue
        q = Queue()
        explored = set()
        q.push((start, []))
        while not q.isEmpty():
            popped = q.pop()
            if popped[0] in explored:
                continue
            explored.add(popped[0])
            if popped[0] == goal:
                return popped[1]
            for successor in getSuccessors(popped[0]):
                if successor[0] not in explored:
                    #push successor, where actions and cost are cumulative on its parent (popped)
                    q.push((successor[0], popped[1] + [successor[1]]))
        return popped[1]
    def getSuccessors(position):
        from game import Actions
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = position
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            walls = currentGameState.getWalls()
            if not walls[nextx][nexty]:
                nextState = (nextx, nexty)
                successors.append((nextState, action))
        return successors

    if currentGameState.isWin():
        return 69000
    if currentGameState.isLose():
        return -69000

    currentPos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    nearestF = 69
    for food in foodList:
        dist = util.manhattanDistance(currentPos, food)
        if (dist < nearestF):
            nearestF = dist

    ghostCount = currentGameState.getNumAgents() - 1
    nearestG = 69
    for ghost in currentGameState.getGhostPositions():
        dist = util.manhattanDistance(currentPos, ghost)
        if (dist < nearestG):
            nearestG = dist

    score = scoreEvaluationFunction(currentGameState)
    scared = False
    for ghost in currentGameState.getGhostStates():
        if ghost.scaredTimer > 0:
            scared = True
    if scared:
        score -= nearestG**2
    score += min(nearestG,3)
    score -= math.sqrt(nearestF)
    capsules = len(currentGameState.getCapsules())
    score -= 100*capsules
    return score
    """
    def mazeDistance(point1, point2):
        return len(bfs(point1, point2))
    def bfs(start, goal):
        from util import Queue
        q = Queue()
        explored = set()
        q.push((start, []))
        while not q.isEmpty():
            popped = q.pop()
            if popped[0] in explored:
                continue
            explored.add(popped[0])
            if popped[0] == goal:
                return popped[1]
            for successor in getSuccessors(popped[0]):
                if successor[0] not in explored:
                    #push successor, where actions and cost are cumulative on its parent (popped)
                    q.push((successor[0], popped[1] + [successor[1]]))
        return popped[1]
    def getSuccessors(position):
        from game import Actions
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = position
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            walls = currentGameState.getWalls()
            if not walls[nextx][nexty]:
                nextState = (nextx, nexty)
                successors.append((nextState, action))
        return successors

    currentPos = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    w = foodGrid.width
    h = foodGrid.height
    hypotenuse = math.sqrt(w*w + h*h)
    closestFood = 69000
    for food in foodGrid.asList():
        dist = mazeDistance(currentPos, food) + manhattanDistance(currentPos, food)
        if dist < closestFood:
            closestFood = dist
    foodScore = (hypotenuse/2)**2/closestFood**2

    ghostScore = 0
    for ghost in currentGameState.getGhostStates():
        dist = util.manhattanDistance(ghost.getPosition(), currentPos)
        timer = ghost.scaredTimer
        if timer == 0:
            ghostScore -= (hypotenuse/2)**3/dist**3
            continue
        ghostScore += (hypotenuse/2)**3/dist**3

    score = 0
    if currentGameState.isWin():
        score += 69000
    elif currentGameState.isLose():
        score -= 69000
    print(foodScore, ghostScore, currentPos)
    return score + foodScore + ghostScore
    """
# Abbreviation
better = betterEvaluationFunction
