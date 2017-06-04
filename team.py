from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util
from game import Directions
import game
from util import nearestPoint
import capture

#------------------------------------------------------------
# Inference Modules
#------------------------------------------------------------
class ParticleFilter(object):
  
  def __init__(self, index, numParticles=300):
      self.index = index
      self.numParticles = numParticles
  
  def reinitialize(self, state):
      self.beliefs = util.Counter()
      for _ in range(self.numParticles):
          choice = random.choice(self.legalPositions)
          self.beliefs[choice] +=  1
      self.beliefs.normalize()
  
  def observe(self, gameState, agent):
      distances      = gameState.getAgentDistances()
      observation    = distances[self.index - 1]
      position       = gameState.getAgentState(agent.index).getPosition()
      possible       = util.Counter()

      for legal in self.legalPositions:
          distance = agent.getMazeDistance(legal, position)
          evidence = gameState.getDistanceProb(distance, observation)
          possible[legal] += evidence * self.beliefs[legal]
      possible.normalize()

      # if we have no clue what is going on, start over
      if possible.totalCount() == 0.0:
          self.reinitialize(gameState)
          return

      self.beliefs = util.Counter()
      for _ in range(self.numParticles):
          self.beliefs[util.sample(possible)] += 1
      self.beliefs.normalize()
    
  def elapseTime(self, gameState):
      pass
      #possible = util.Counter()
      #for past in self.legalPositions:
      #    if past not in self.beliefs: continue
      #    pprob = self.beliefs[past]
      #    state = self.setEnemyPosition(gameState, past)
      #    distribution = self.getPositionDistribution(state)
      #    for future, fprob in distribution.items():
      #        possible[future] += self.beliefs[past] * fprob

      #self.beliefs = possible
      #self.beliefs.normalize()

  def setEnemyPosition(self, gameState, ghostPosition):
      conf = game.Configuration(ghostPosition, game.Directions.STOP)
      gameState.data.agentStates[self.index] = game.AgentState(conf, False)
      return gameState

#------------------------------------------------------------
# Singleton Inference Module
#------------------------------------------------------------
class EnemyFinder(object):

    instance = None    
    def __init__(self, state, agent):
        self.enemies = agent.getOpponents(state)
        self._legal = [p for p in state.getWalls().asList(False) if p[1] > 1]   
        self.models = dict((e, ParticleFilter(e)) for e in self.enemies)
        for model in self.models.values():
            model.legalPositions = self._legal
            model.reinitialize(state)

    def observe(self, state, agent):
        for model in self.models.values():
            model.observe(state, agent)

    def find(self, state, enemy):
        # first try absolute
        agent = state.getAgentState(enemy)
        position = agent.getPosition()
        if position != None: return position

        # then try fuzzy
        agent = self.models[enemy]
        return agent.beliefs.argMax()
        

#------------------------------------------------------------
# Team Creation
#------------------------------------------------------------
def createTeam(firstIndex, secondIndex, isRed,
               first = 'BothReflexAgent', second = 'BothReflexAgent'):
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
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

#------------------------------------------------------------
# Agent
#------------------------------------------------------------
class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    if EnemyFinder.instance == None:
        EnemyFinder.instance = EnemyFinder(gameState, self)
    EnemyFinder.instance.observe(gameState, self)

    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    return random.choice(bestActions)

    #my_team = self.getTeam(gameState)

    #def min_value(state, depth, agent):
    #    agent = 3 if agent < 0 else agent
    #    moves = state.getLegalActions(agent)
    #    if Directions.STOP in moves:
    #        moves.remove(Directions.STOP)
    #    if len(moves) == 0 or depth == 0:
    #        return (self.evaluate(state, 'STOP'), 'STOP')
    #    next_value = max_value if agent in my_team else min_value
    #    successors = ((state.generateSuccessor(agent, m), m) for m in moves)
    #    if agent == self.index: depth -= 1
    #    return min((next_value(s, depth, agent - 1)[0], m) for s,m in successors)

    #def max_value(state, depth, agent):
    #    moves = state.getLegalActions(agent)
    #    if Directions.STOP in moves:
    #        moves.remove(Directions.STOP)
    #    if len(moves) == 0 or depth == 0:
    #        return (self.evaluate(state, 'STOP'), 'STOP')
    #    successors = ((state.generateSuccessor(agent, m), m) for m in moves)
    #    return max((min_value(s, depth, agent - 1)[0], m) for s,m in successors)

    #return max_value(gameState, 3, self.index)[1]

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else: return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    # cache this
    bads = (-1,-1) if self.red else (1000, 1000)
    pacs = [(i, gameState.getAgentState(i)) for i in self.getTeam(gameState)]
    redo = any(gameState.getInitialAgentPosition(i) == p.getPosition() for i, p in pacs)

    # -----------------------------------------------------
    # only switch if
    # 1. we have just started (to set initial mode)
    # 2. if someone got killed (to restart attack faster)
    # -----------------------------------------------------
    #if redo or not hasattr(self, 'mode'):
    #    team = [(p.getPosition(), i) for i,p in pacs]
    #    compare = min if self.red else max
    #    player = compare((p[0], i) for p, i in team)
    #    self.mode = 'defense' if player[1] == self.index else 'offense'
    self.mode = 'offense'

    features = self.getFeatures(gameState, action)
    weights  = self.getWeights(gameState, action)
    return features * weights

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
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    features['successorScore'] = self.getScore(successor)

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    if len(invaders) > 0:
      dist = min([self.getMazeDistance(myPos, a.getPosition()) for a in invaders])
      features['ghostDistance'] = 1 if dist < 4 else  0

    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1, 'ghostDistance':-100}

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
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}


class BothReflexAgent(ReflexCaptureAgent):
  ''' An agent that switches its role based on the current situation
  '''

  def getOffenseFeatures(self, gameState, action):
    ''' Retrieve the values for the offensive features
    :param gameState: The current game state to evaluate
    :param action: The selected action to take
    :returns: The selected feature values
    '''
    # -----------------------------------------------------
    # initialize settings
    # -----------------------------------------------------
    features  = util.Counter()
    successor = self.getSuccessor(gameState, action)
    state     = successor.getAgentState(self.index)
    position  = state.getPosition()
    features['successorScore'] = self.getScore(successor)
    features['stop']           = (action == Directions.STOP)

    # -----------------------------------------------------
    # Computes distance to defenders we can see
    # -----------------------------------------------------
    enemies  = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    print [EnemyFinder.instance.find(gameState, i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    if len(invaders) > 0 and state.isPacman:
      dist = min([self.getMazeDistance(position, a.getPosition()) for a in invaders])
      features['defenderDistance'] = -100 if dist < 2 else dist
    else: features['defenderDistance'] = 100 # we got a power pill

    # -----------------------------------------------------
    # Computes distance to power pellets
    # -----------------------------------------------------
    isScared  = any(successor.getAgentState(i).scaredTimer > 0 for i in self.getOpponents(successor))
    powerList = self.getCapsules(successor)
    if (len(powerList) > 0) and not isScared:
      distances = (self.getMazeDistance(position, power) for power in powerList)
      features['distanceToPower'] = min(distances)
    else: features['distanceToPower'] = 0
    # don't negative weight yet

    # -----------------------------------------------------
    # computes distance to team mate
    # -----------------------------------------------------
    partners = [successor.getAgentState(i) for i in self.getTeam(successor)]
    if len(partners) > 0:
      dist = max([self.getMazeDistance(position, a.getPosition()) for a in partners])
      features['partnerDistance'] = dist
    else: features['partnerDistance'] = 0

    
    # -----------------------------------------------------
    # Computes distance to invaders we can see
    # -----------------------------------------------------
    enemies  = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    if len(invaders) > 0 and not state.isPacman:
      dist = min([self.getMazeDistance(position, a.getPosition()) for a in invaders])
      features['ghostDistance']     = dist
    else: features['ghostDistance'] = 0 # we are attacking

    # -----------------------------------------------------
    # Compute distance to the nearest food
    # -----------------------------------------------------
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0:
      distances = (self.getMazeDistance(position, food) for food in foodList)
      features['distanceToFood'] = min(distances)
    return features

  def getDefenseFeatures(self, gameState, action):
    ''' Retrieve the values for the defensive features
    :param gameState: The current game state to evaluate
    :param action: The selected action to take
    :returns: The selected feature values
    '''
    # -----------------------------------------------------
    # initialize settings
    # -----------------------------------------------------
    features  = util.Counter()
    successor = self.getSuccessor(gameState, action)
    state     = successor.getAgentState(self.index)
    position  = state.getPosition()

    # -----------------------------------------------------
    # Computes whether we're on defense (1) or offense (0)
    # -----------------------------------------------------
    features['onDefense'] = not state.isPacman

    # -----------------------------------------------------
    # Computes distance to invaders we can see
    # -----------------------------------------------------
    enemies  = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      distances = (self.getMazeDistance(position, a.getPosition()) for a in invaders)
      features['invaderDistance'] = min(distances)

    # -----------------------------------------------------
    # action penalties
    # -----------------------------------------------------
    reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    features['stop']    = (action == Directions.STOP)
    features['reverse'] = (action == reverse)

    return features

  def getFeatures(self, state, action):
    ''' Retrieve the values for the selected features
    :param state: The current game state to evaluate
    :param action: The selected action to take
    :returns: The selected feature values
    '''
    if self.mode == 'offense':
        return self.getOffenseFeatures(state, action)
    return self.getDefenseFeatures(state, action)

  def getWeights(self, state, action):
    ''' Retrieve the weights for the selected features
    :param state: The current game state to evaluate
    :param action: The selected action to take
    :returns: The selected feature weights
    '''
    if self.mode == 'offense':
        return {
            'successorScore':  100, # eating a pellet bonus
            'distanceToFood':   -1, # distance to closest food penalty
            'defenderDistance': -1, # increase distance from defender
            'ghostDistance':     0, # penalty for attacking ghost
            'stop':           -100, # penalty for standing still
            'distanceToPower': -10, # get that power pellet
            'parterDistance':    0, # keep the partners spread
        }
    else:
        return {
            'numInvaders':   -1000, # penalty for invading ghosts
            'onDefense':       100, # ensure we stay on defense
            'invaderDistance': -10, # penalty for attacking ghost
            'stop':           -100, # penalty for standing still
            'reverse':          -2, # penalty for reversing (starvation)
        }

class BlockerReflexAgent(ReflexCaptureAgent):
  ''' An agent that switches its role based on the current situation
  '''

  def getWeights(self, state, action):
      return { 'blocked' : -100 }

  def getFeatures(self, state, action):
      blocker   = (17, 8) if not self.red else (16, 9)
      features  = util.Counter()
      successor = self.getSuccessor(state, action)
      state     = successor.getAgentState(self.index)
      position  = state.getPosition()
      distance  = self.getMazeDistance(position, blocker)
      features['blocked'] = distance
      return features
