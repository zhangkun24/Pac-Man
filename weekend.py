from captureAgents import CaptureAgent
import random, time, util
from util import nearestPoint
from game import Directions
import game


def createTeam(firstIndex, secondIndex, isRed,
               first = 'UpperAgent', second = 'LowerAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.
  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]



class BaseAgent(CaptureAgent):
    #register initial
    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        
        # get the teammate index
        self.teamMateIndices = self.getTeam(gameState)
        self.teamMateIndices.remove(self.index)
        self.lastInfo = {}
        self.lastInfo['trueInvaders'] = 0

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor
    
    # state of the agent
    def isPac(self, gameState):

        return gameState.getAgentState(self.index).isPacman

    # get ghost states and  position
    def ghostStatPos(self, gameState):
        myPos = gameState.getAgentState(self.index).getPosition()
        opponentStates = [gameState.getAgentState(g) for g in self.getOpponents(gameState)]
        gStatPos = [{'scaredTimer': g.scaredTimer, 'pos' :g.getPosition()}  for g in opponentStates if 
                g.getPosition() !=None and not g.isPacman]
        #ghostPositions = [g.getPosition() for g in opponentStates and not g.isPacman and g.getPosition() != None]
        #distanceToGhosts = [self.getMazeDistance(myPos, gPos) for gPos in ghostPositions]
        return gStatPos
    

    def offensiveFeatures(self, gameState, action):
        myPos = gameState.getAgentState(self.index).getPosition()
        matePos = gameState.getAgentState(self.teamMateIndices[0]).getPosition()
        features = util.Counter()

        # feature No3. distance to ghost, No4. dist to panic ghost
        distToGhost = 999999
        distToScaredGhost  = 999999
        mateToGhost = 999999
        numScared = 0
        gStatPos = self.ghostStatPos(gameState)
        if gStatPos != None and len(gStatPos) > 0:
            for i in range(len(gStatPos)):
                if gStatPos[i]['scaredTimer'] == 0:
                    distToGhost = min(distToGhost, self.getMazeDistance(myPos, gStatPos[i]['pos']))
                    mateToGhost = min(mateToGhost, self.getMazeDistance(matePos, gStatPos[i]['pos']))
                else:
                    numScared += 1
                    distToScaredGhost = min(distToScaredGhost, self.getMazeDistance(myPos, gStatPos[i]['pos']))
        #print "dist to scared: ", distToScaredGhost
        #print "dist to ghost: ", distToGhost
        features['distanceToGhost'] = 1.0/distToGhost  if distToGhost <= 4 else  0
        features['getScaredGhost'] =  1.0/distToScaredGhost if distToScaredGhost <=4 else 0
        features['numScared'] = numScared    # tryingn to take out the scared ghost but only when they are in 2 dist
    
        #feature No.5, get capsules
        numCaps = len(self.getCapsules(gameState))
        if numCaps > 0 and (mateToGhost <= 5 or distToGhost <= 5):
                self.toGetCap = True
                distToCaps = [self.getMazeDistance(myPos, capPos) for capPos in self.getCapsules(gameState)]
                minDistToCaps = min(distToCaps)
                #print '----------------------------------------------------'
                #print 'mate To ghost: ', mateToGhost
                #print 'my dist to cap: ', distToCaps 
                features['distToCap'] = 1.0 /minDistToCaps
        else:
            features['distToCap'] = 11
        """
        # feat
        capsules = self.getCapsules(gameState)
        if(len(capsules) > 0):
          minCapsuleDist = min([self.getMazeDistance(myPos, capsule) for capsule in capsules])
        else:
          minCapsuleDist = .1
          
        features['distToCap'] =  1.0 / minCapsuleDist
        """
        # feature No6 and 7, novaluable move
        #Don't want to do these things
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    # weights function
    def offensiveWeights(self, gameState, action):
        return {'successorScore' :100, 'distanceToFood' : -1, 'distanceToGhost': -400, 'getScaredGhost': 20,
                'numScared': -100, 'distToCap': 10, 'stop':-1000, 'reverse':-30}



    #function to return deffensive features
    def defensiveFeatures(self, gameState, action):
        myPos = gameState.getAgentState(self.index).getPosition()
        isScared = gameState.getAgentState(self.index).scaredTimer != 0
        #matePos = gameState.getAgentState(self.teamMateIndices[0]).getPosition()
        features = util.Counter()
        features['numInvaders'] = 0
        features['invaderDistance'] = 0
        # Computes distance to invaders we can see
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        invaders = [a for a in enemies if a.getPosition() != None and self.getMazeDistance(myPos, a.getPosition()) <= 3]
        trueInvaders = [a for a in enemies if a.getPosition() != None and a.isPacman
                and self.getMazeDistance(myPos, a.getPosition()) <= 3]
        #features['numInvaders'] = len(invaders)
        dists = 999999
        if len(invaders) > 0 and dists > 0 and not isScared:
          dists = min(dists, min([self.getMazeDistance(myPos, a.getPosition()) for a in invaders]))
          features['invaderDistance'] = 1.0/dists
        if len(trueInvaders) < self.lastInfo['trueInvaders'] :
            features['invaderDistance'] = 1000
        self.lastInfo['trueInvaders'] = len(trueInvaders) 
        return features

    #defensive weights
    def defensiveWeights(self, gameState, action):
        return {'successorScore' :100, 'distanceToFood' : -2, 'invaderDistance': 2.5} 
                

    # split food to half
    def foodSplit(self, gameState):
        foodToEat = self.getFood(gameState).asList()
        foodByY = sorted(foodToEat, key=lambda x: x[1])
        upperFood = foodByY[: len(foodByY)/2]
        lowerFood = foodByY[len(foodByY)/2 :]
        return {'upperFood': upperFood, 'lowerFood': lowerFood}


class UpperAgent(BaseAgent):
    # register initial
    def registerInitialState(self, gameState):
        BaseAgent.registerInitialState(self, gameState)

    #choose action
    def chooseAction(self, gameState):
        
        start = time.time() 
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]
        #print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        
        return random.choice(bestActions)
    
    # evaluate function
    def evaluate(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights  = self.getWeights(gameState, action)
        return features * weights
    

    # get feature function
    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        # feature No1. successorScore
        features['successorScore'] = self.getScore(successor)

        # feature No2. distanceToFood
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        foodList = self.foodSplit(successor)['upperFood']
        if self.isPac(successor) and not successor.getAgentState(self.teamMateIndices[0]).isPacman:
            foodList = self.getFood(successor).asList()
        if len(foodList) > 0:
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        ghostRrelated = None
        if self.isPac(successor):
            #print 'off-----'
            # feature No3 and No4
            ghostRelated = self.offensiveFeatures(successor, action)
        else:
            #print 'def--------'
            ghostRelated = self.defensiveFeatures(successor, action)
        if ghostRelated:
            features = features + ghostRelated

        return features


    # weights function
    def getWeights(self, gameState, action):
        successor = self.getSuccessor(gameState,action)
        if self.isPac(successor):
            return self.offensiveWeights(successor, action)
        else:
            return self.defensiveWeights(successor, action)



class LowerAgent(BaseAgent):
    # register initial
    def registerInitialState(self, gameState):
        BaseAgent.registerInitialState(self, gameState)
        self.type = 'offensive'


    #choose action
    def chooseAction(self, gameState):
        
        actions = gameState.getLegalActions(self.index)
        values = [self.evaluate(gameState, a) for a in actions]
        #print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]
        
        return random.choice(bestActions)
    
    # evaluate function
    def evaluate(self, gameState, action):
        features = self.getFeatures(gameState, action)
        weights  = self.getWeights(gameState, action)
        return features * weights
    

    # get feature function
    def getFeatures(self, gameState, action):
        features = util.Counter()
        #No1 and No2 are basic features
        #feature No1. successorScore
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        
        #feature No2. distance to food
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        foodList = self.foodSplit(successor)['lowerFood']
        if self.isPac(successor) and not successor.getAgentState(self.teamMateIndices[0]).isPacman:
            foodList = self.getFood(successor).asList()
        if len(foodList) > 0:
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        #feature No3. distance to ghost, run away from ghost 
        ghostRrelated = None
        if self.isPac(successor):
            #print 'off-----'
            # feature No3 and No4
            ghostRelated = self.offensiveFeatures(successor, action)
        else:
            #print 'def------'
            ghostRelated = self.defensiveFeatures(successor, action)
        if ghostRelated:
            features = features + ghostRelated
        return features


    # weights function
    def getWeights(self, gameState, action):
        successor = self.getSuccessor(gameState,action)
        if self.isPac(successor):
            return self.offensiveWeights(successor, action)
        else:
            return self.defensiveWeights(successor, action)





