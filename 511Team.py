
# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util
from game import Directions
import game
from util import nearestPoint
import util

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'CleverReflexAgent', second = 'DefensiveReflexAgent'):
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
  return [eval(first)(firstIndex), eval(first)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    return random.choice(bestActions)

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

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    #print 'self.index',  self.index
    #print 'features',  features 
    #print 'weights',  weights
    #print 'action', action
    #print "the score is" 
    #print features * weights
    return features * weights

    """
    features {'getAggressive': 0, 'capsule': 0, 'numFood': 2, 'instantFood': 1000, 'minDistanceToFood': 1}
    weights {'ghost': 5, 'getAggressive': 1, 'capsule': 1, 'instantFood': 1, 'minDistanceToFood': -1, 'numFood': -10, 'stop': -1, 'instantGhost': -1}
    """

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.isOffense = True

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """

    features = self.getFeatures(gameState, action)

    #print 'features',  features 

    return -0.1* features['minDistanceToFood'] + 2*features['ghost'] - 2*features['numFood']
   
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    # Computes distance to invaders we can see
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    ghostsPos = []
    for enemyI in self.getOpponents(gameState):
      ghostPos = gameState.getAgentPosition(enemyI)
      if ghostPos != None:
        ghostsPos.append((enemyI,ghostPos))

    if len(ghostsPos) > 0:
      for ghost, pos in ghostsPos:
        if ghost <=1: ##the ghost is close but it is offensive, do not be afraid
          if myState.isPacman == True:
            pass
          else:
            features['ghost'] -= 10*self.getMazeDistance(myPos,pos)
        else:  ##the ghost is close, it is defensive, watch out
          features['ghost'] += 10*self.getMazeDistance(myPos,pos)

    dists = 999

    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    features['numFood'] = len(foodList)
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'minDistanceToFood': -0.1,'numFood':-2,'ghost':2}

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)


    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -100000, 'onDefense': 100, 'invaderDistance': -100*2, 'stop': -100, 'reverse': -2}


class CleverReflexAgent(ReflexCaptureAgent):
  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.ready = False
    myTeam = self.getTeam(gameState)
    self.team = {}
    self.team[myTeam[0]] = 1
    self.team[myTeam[1]] = 2

    startx = gameState.getWalls().width/2
    starty = gameState.getWalls().height/2
    startPos = []
    if self.getTeam(gameState)[0]%2 != 0:
      startx -=1
    self.temStartPoint =(startx,starty)
    minDist = 99999
    myPos = gameState.getAgentState(self.index).getPosition()
    while starty>=0:
      if gameState.hasWall(startx,starty) == False:
        dist = self.getMazeDistance(myPos,(startx,starty))
        if dist < minDist:
          self.Bstart = (startx,starty)
          minDist = dist
      starty -=1

    startx,starty = self.temStartPoint
    minDist = 99999
    for i in xrange(gameState.getWalls().height-starty):
      if gameState.hasWall(startx,starty) == False:
        dist = self.getMazeDistance(myPos,(startx,starty))
        if dist < minDist:
          self.Astart = (startx,starty)
          minDist = dist
      starty +=1
    
    self.start = (16,15)
    self.status = None
    basePoint = gameState.getAgentState(myTeam[1]).getPosition()
    x = basePoint[0]
    y = basePoint[1]
    self.opponentStatus = {}
    if self.getTeam(gameState)[0]%2 != 0:   ## set the origin point,the team is blue
      self.teamName = "blue"
      if self.index == 1:
        self.basePoint = [(x,y-1),(float(x),float(y-1))]
      elif self.index == 3:
        self.basePoint = [(x,y),(float(x),float(y))]
      self.opponentStatus[0] = False
      self.opponentStatus[2] = False
    else:
      self.teamName = "red"
      if self.index == 0:
        self.basePoint = [(x,y+1),(float(x),float(y+1))]
      elif self.index == 2:
        self.basePoint = [(x,y),(float(x),float(y))]
      self.opponentStatus[1] = False
      self.opponentStatus[3] = False

  def getFeatures(self, gameState, action):
    #print self.status
    myCurrentState  = gameState.getAgentState(self.index)
    scaredTime = myCurrentState.scaredTimer
    if scaredTime > 1 and myCurrentState.isPacman == False:
      self.status = "offense"
      return self.getOffenseFeatures(gameState,action) ##althought you are a ghost, but you are scared, so you would better avoid the pacman and go to offense

    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    if self.getPreviousObservation()!= None:
      currentPosition = gameState.getAgentState(self.index).getPosition()
      lastPosition = gameState.getAgentState(self.index).getPosition()
      if currentPosition == self.basePoint[0] or currentPosition == self.basePoint[1]:
        print "Wo bei chi diao le!"
        print self.ready
        self.ready = False
        print self.ready

    ghosts = []
    for opponent in self.getOpponents(gameState):
      ghostPos = gameState.getAgentPosition(opponent)
      if ghostPos != None:
        ghosts.append((opponent,ghostPos))
    if len(ghosts) > 0:
      for opponent,ghostPos in ghosts:
        if myState.isPacman == False and self.getMazeDistance(myPos,ghostPos)<6:
          self.status = "defense"
          return self.getDefenseFeatures(gameState,action)
    if self.ready == False:
      self.status = "ready"
      return self.getReadyFeatures(gameState, action)
    else:
      self.status = "offense"
      return self.getOffenseFeatures(gameState,action)

  def getReadyFeatures(self, gameState, action):
    features = util.Counter()

    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    if self.team[self.index] == 1:
      self.start = self.Astart
    else:
      self.start = self.Bstart
    startDist = self.getMazeDistance(myPos, self.start)
    features['startDist'] = startDist
    if myPos == self.start:
      features["ready"] = 1
      self.ready = True
    if action == "Stop":
      features["stop"] = 9999999
    return features

  def getTeammate(self,gameState):
    if self.index == 0:
      return 2
    elif self.index == 1:
      return 3
    elif self.index == 2:
      return 0
    elif self.index == 3:
      return 1

  def getOffenseFeatures(self, gameState, action):
    features = util.Counter()
    print 'offenseFeatures: ', features
    myCurrentState  = gameState.getAgentState(self.index)
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    '''print myPos'''

    x = int(myPos[0])
    y = int(myPos[1])

    if len(self.getCapsules(gameState)):
      if (x,y) == self.getCapsules(gameState)[0]:
        print "eat capsule!!"
        features['capsule'] = 100000
      else:
        features['capsule'] = 0
    else:
      features['capsule'] = 0
 
    if self.getFood(gameState)[x][y] == True and self.getFood(successor)[x][y] == False:
      features['instantFood'] = 1000
    dists = 999
    ghosts = []
    for opponent in self.getOpponents(gameState):
      ghostPos = gameState.getAgentPosition(opponent)
      if ghostPos != None:
        ghosts.append((opponent,ghostPos))
    if len(ghosts) > 0:
      for opponent,ghostPos in ghosts:
        if self.getMazeDistance(myPos,ghostPos)<=6:
          if myPos == ghostPos:
            features['instantGhost'] = 100000
          features['ghost'] += self.getMazeDistance(myPos,ghostPos)
        if myPos == ghostPos and gameState.getAgentState(opponent).scaredTimer>1 and self.opponentStatus[opponent] == False:
          print gameState.getAgentState(opponent).scaredTimer
          print "I am crazy!!!!"
          features['getAggressive'] = 10000000
          self.opponentStatus[opponent] = True
        else:
          features['getAggressive'] = 0
    else:
      features['ghost'] = 7

    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    features['numFood'] = len(foodList)
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minFoodDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['minDistanceToFood'] = minFoodDistance
    if action == "Stop":
      features["stop"] = 9999999
    return features

  def getDefenseFeatures(self, gameState, action):

    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    if self.status == "offense":
      return self.getOffenseWeights(gameState,action)
    elif self.status == "defense":
      return self.getDefenseWeights(gameState,action)
    elif self.status == "ready":
      return self.getReadyWeights(gameState,action)

  def getOffenseWeights(self, gameState, action):
    return {'minDistanceToFood': -1, 'numFood': -10, 'stop':-1, 'ghost':5, 'instantFood':1,'capsule':1,'getAggressive':1,'instantGhost':-1}

  def getDefenseWeights(self, gameState, action):
    '''return {'numInvaders': -100000, 'onDefense': 100, 'invaderDistance': -100**2, 'stop': -100, 'reverse': -2, 'stop':-1,'capsule':1}'''
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}

  def getReadyWeights(self, gameState, action):
    return {'startDist': -1, 'ready': 500,'stop':-1}

