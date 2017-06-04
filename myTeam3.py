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
import capture

# Awesome, global variableness
bearassedInfAgent = None


#################
# Team creation #
#################

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

class BothReflexAgent(ReflexCaptureAgent):
  ''' An agent that switches its role based on the current situation
  '''
  
  def getOffenseFeatures(self, gameState, action):
    ''' Retrieve the values for the offensive features
    
    :param gameState: The current game state to evaluate
    :param action: The selected action to take
    :returns: The selected feature values
    '''
    global bearassedInfAgent
    
    # -----------------------------------------------------
    # initialize settings
    # -----------------------------------------------------
    features  = util.Counter()
    successor = self.getSuccessor(gameState, action)
    state     = successor.getAgentState(self.index)
    position  = state.getPosition()
    features['successorScore'] = self.getScore(successor)
    features['stop']           = (action == Directions.STOP)
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    team = [successor.getAgentState(i) for i in self.getTeam(successor)]
     
    # Only one of our agents needs to initialize this common helper agent
    if bearassedInfAgent == None:
        print "DEBUG: Initializing Inference Agent"
        bearassedInfAgent = BearassedInfAgent()
        bearassedInfAgent.registerInitialState(successor, team, enemies, self.getTeam(successor), self.getOpponents(successor)) 
        
    # -----------------------------------------------------
    # Updates beliefs about the enemy pacmen/ghosts
    # -----------------------------------------------------
    bearassedInfAgent.getDistribution(successor)
    
    # -----------------------------------------------------
    # Computes distance to defenders we can see
    # -----------------------------------------------------
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
            'defenderDistance':  1, # increase distance from defender
            'ghostDistance':     0, # penalty for attacking ghost
            'stop':           -100, # penalty for standing still
            'distanceToPower':  -1, # get that power pellet
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
        
# Blatently stolen from project 4 and then haxxored
class BearassedInfAgent:
  "An agent that tracks and displays its beliefs about enemy positions."
  
  def __init__( self ):
    # I have lazily selected the one and only inference type available
    # Also we only need one, since the particle filter will track both enemy agents
    self.inferenceModules = [BearassedParticleFilter(700)]
  
  def registerInitialState(self, gameState, team, enemies, teamIndecies, enemyIndecies):
    "Initializes beliefs and inference modules"
    import __main__
    self.display = __main__._display
    legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    for inference in self.inferenceModules: inference.initialize(gameState, legalPositions)
    self.enemyBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
    self.firstMove = True
    
    # Setup the agents
    i = 0
    for inf in self.inferenceModules:
        for t in team:
            inf.addTeamAgent(t,teamIndecies[i])
        for e in enemies:
            inf.addEnemyAgent(e,enemyIndecies[i])
        i += 1
    
  def observationFunction(self, gameState):
    "Removes the enemy states from the gameState"
    agents = gameState.data.agentStates
    gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
    return gameState

  # This was specifically created for the contest
  def getDistribution(self, gameState):
    "Updates beliefs, then returns a distribution for each enemy model"
    for index, inf in enumerate(self.inferenceModules):
        if not self.firstMove: 
            inf.elapseTime(gameState)
        self.firstMove = False
        inf.observeState(gameState)
        self.enemyBeliefs[index] = inf.getBeliefDistribution()
        
    self.display.updateDistributions(self.enemyBeliefs)
    
    return inf.getBeliefDistribution()
    
  def getAction(self, gameState):
    "Updates beliefs, then chooses an action based on updated beliefs."
    for index, inf in enumerate(self.inferenceModules):
      if not self.firstMove: inf.elapseTime(gameState)
      self.firstMove = False
      inf.observeState(gameState)
      self.enemyBeliefs[index] = inf.getBeliefDistribution()
    return self.chooseAction(gameState)

  def chooseAction(self, gameState):
    "By default, a Agent just stops.  This should be overridden."
    return Directions.STOP
      
        
# Borrowed from project 4 and then haxxored
# This is shared between both of our team's agents
class BearassedParticleFilter:
  "BearassedParticleFilter tracks a joint distribution over tuples of all enemy agent positions."

  def __init__(self, numParticles=600):
    print "DEBUG: Initializing the particle filter with", numParticles,"particles"
    self.setNumParticles(numParticles)
  
  def setNumParticles(self, numParticles):
    self.numParticles = numParticles
  
  def initialize(self, gameState, legalPositions):
    "Stores information about the game, then initializes particles."
    self.numEnemies = 2 # hardcoded, magical number
    self.enemyAgents = []
    self.teamAgents = []
    self.enemyIndecies = []
    self.teamIndecies = []
    self.legalPositions = legalPositions
    self.initializeParticles()
    
  def initializeParticles(self):
    "Initializes particles randomly.  Each particle is a tuple of enemy positions. Use self.numParticles for the number of particles"
    self.particles = []
    for i in range(0,self.numParticles):
        enemyPositions = []
        for j in range(0,self.numEnemies):
            enemyPositions.append(random.choice(self.legalPositions))
        self.particles.append(tuple(enemyPositions))

  def addTeamAgent(self, agent, index):
    "Each of our team's agent is registered separately and stored (in case they are different)."
    self.teamAgents.append(agent)
    self.teamIndecies.append(index)
    print "team agent=",agent,index
        
  def addEnemyAgent(self, agent, index):
    "Each enemy agent is registered separately and stored (in case they are different)."
    self.enemyAgents.append(agent)
    self.enemyIndecies.append(index)
    print "enemy agent=",agent, index
    
  def elapseTime(self, gameState):
    print "elapseTime"
    """
    Samples each particle's next state based on its current state and the gameState.
    """
    newParticles = []
    for oldParticle in self.particles:
        prevEnemyPositions = list(oldParticle) # A list of enemy positions
        newParticle = []
        for i in range(self.numEnemies):
            newPosDist = getPositionDistributionForEnemy(setEnemyPositions(gameState, prevEnemyPositions),
                                                         i, self.enemyAgents[i])
            newParticle.append(util.sample(newPosDist))
        newParticles.append(tuple(newParticle))
    self.particles = newParticles

  def getJailPosition(self, i):
    return (2 * i + 1, 1);
  
  def observeState(self, gameState, absolute=False):
    """
    Resamples the set of particles using the likelihood of the noisy observations.
    absolute = True  - indicates that this is an observation generated by seeing a food eatten
                       when that happens we know for sure where the enemy pacman is
                       TODO: Not clear on how we'll know which enemy it is...
    """ 
    #pacmanPosition = gameState.getPacmanPosition()
    teamPositions = []
    for i in self.teamIndecies:
        teamPositions.append(gameState.getAgentState(i).getPosition())
    
    noisyDistances = []
    for i in self.enemyIndecies:
        noisyDistances.append(gameState.getAgentState(i).getPosition())

    print "team positions", teamPositions
    print "enemy positions",noisyDistances   

    # loop over each particle to gather the weights
    # likelihood = P(E | G1) * P(E | G2) * ...
    # store in dist 
    dist = util.Counter()
    newParticles = [] # this is needed to update particles if enemies go to jail
    for particle in self.particles:
        newParticle = []
        prob = 1
        for i in range(self.numEnemies):
            # Sent this ghost to jail --
            # TODO - this was jail but now maybe we just cannot hear the enemy
            if noisyDistances[i] == None:
                newParticle.append(self.getJailPosition(i))
            else:
                trueDistance = util.manhattanDistance(particle[i], pacmanPosition)
                prob *= gameState.getDistanceProb(trueDistance, noisyDistances[i])
                newParticle.append(particle[i])
        dist[tuple(newParticle)] += prob
            
        newParticles.append(tuple(newParticle))
        
    self.particles = tuple(newParticles)
    
    # Check for corner case where all particles are 0
    if dist.totalCount() == 0:
        self.initializeParticles()
    else:
        # resample from our weighted distribution
        newParticles = [] # reuse to generate new particles based on dist
        for i in range(0,self.numParticles):
            sample = util.sample(dist)
            newParticles.append(tuple(sample))
        self.particles = newParticles
    
  def getBeliefDistribution(self):
    dist = util.Counter()
    for part in self.particles: dist[part] += 1
    dist.normalize()
    return dist

# One JointInference module is shared globally across instances of MarginalInference 
#bearassedJointInference = BearassedParticleFilter()

def getPositionDistributionForEnemy(gameState, enemyIndex, agent):
  """
  Returns the distribution over positions for a enemy, using the supplied gameState.
  """

  # index 0 is pacman, but the students think that index 0 is the first enemy.
  enemyPosition = gameState.getEnemyPosition(enemyIndex+1) 
  actionDist = agent.getDistribution(gameState)
  dist = util.Counter()
  for action, prob in actionDist.items():
    successorPosition = game.Actions.getSuccessor(enemyPosition, action)
    dist[successorPosition] = prob
  return dist
  
def setEnemyPositions(gameState, enemyPositions):
  "Sets the position of all enemies to the values in enemyPositionTuple."
  for index, pos in enumerate(enemyPositions):
    conf = game.Configuration(pos, game.Directions.STOP)
    gameState.data.agentStates[index + 1] = game.AgentState(conf, False)
  return gameState  
