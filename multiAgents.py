from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
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
    Your minimax agent (question 3)
    """

    def getAction(self, gameState):
        
        def Minimax_Value(self,gameState,agentIndex=0):
            if self.depth<=0 or gameState.isWin() or gameState.isLose():    # 终止条件
                return self.evaluationFunction(gameState)
            actions=gameState.getLegalActions(agentIndex)
            if agentIndex==0:
                self.depth-=1
                v=-1e5
                for action in actions:
                    nextState=gameState.generateChild(agentIndex,action)
                    v=max(v,Minimax_Value(self,nextState,agentIndex+1))
                return v
            else:
                v=1e5
                for action in actions:
                    nextState=gameState.generateChild(agentIndex,action)
                    v=min(v,Minimax_Value(self,nextState,(agentIndex+1)%(gameState.getNumAgents())))    #取模，构成循环
                return v
       
       
        max_act=gameState.getLegalActions(0)[0]     #设置一个默认动作
        v=-1e5
        for act in gameState.getLegalActions(0):
            nextState=gameState.generateChild(0,act)
            cur_v=Minimax_Value(self,nextState,1)
            if v<cur_v:
                max_act=act
                v=cur_v
        return max_act
                
        
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        we assume ghosts act in turn after the pacman takes an action
        so your minimax tree will have multiple min layers (one for each ghost)
        for every max layer

        gameState.generateChild(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state

        self.evaluationFunction(state)
        Returns pacman SCORE in current state (useful to evaluate leaf nodes)

        self.depth
        limits your minimax tree depth (note that depth increases one means
        the pacman and all ghosts has already decide their actions)
        """
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def Minimax_Value(self,gameState,alpha,beta,agentIndex=0):
            if self.depth<=0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            actions=gameState.getLegalActions(agentIndex)
            if agentIndex==0:
                self.depth-=1
                v=-1e5
                for action in actions:
                    nextState=gameState.generateChild(agentIndex,action)
                    v=max(v,Minimax_Value(self,nextState,alpha,beta,agentIndex+1))
                    if v>=beta:return v
                    alpha=max(alpha,v)
                return v
            else:
                v=1e5
                for action in actions:
                    nextState=gameState.generateChild(agentIndex,action)
                    v=min(v,Minimax_Value(self,nextState,alpha,beta,(agentIndex+1)%(gameState.getNumAgents())))
                    if v<=alpha:return v
                    beta=min(beta,v)
                return v
       
        max_act=gameState.getLegalActions(0)[0]     #设置一个默认动作
        v=-1e5
        for act in gameState.getLegalActions(0):
            nextState=gameState.generateChild(0,act)
            cur_v=Minimax_Value(self,nextState,-1e5,1e5,1)
            if v<cur_v:
                max_act=act
                v=cur_v
        return max_act
        util.raiseNotDefined()
