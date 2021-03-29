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
        def Minimax_Value(self,gameState,depth,agentIndex=0):
            if depth<=0 or gameState.isWin() or gameState.isLose():    # 终止条件
                return self.evaluationFunction(gameState)
            actions=gameState.getLegalActions(agentIndex)
            if agentIndex==0:
                v=-1e5
                for action in actions:
                    nextState=gameState.generateChild(agentIndex,action)
                    v=max(v,Minimax_Value(self,nextState,depth-1,agentIndex+1))
                return v
            else:
                v=1e5
                for action in actions:
                    nextState=gameState.generateChild(agentIndex,action)
                    v=min(v,Minimax_Value(self,nextState,depth,(agentIndex+1)%(gameState.getNumAgents())))    #取模，构成循环
                return v
       
        max_act=gameState.getLegalActions(0)[random.randint(0,len(gameState.getLegalActions(0))-1)]
        value=-1e5
        for act in gameState.getLegalActions(0):
            nextState=gameState.generateChild(0,act)
            cur_v=Minimax_Value(self,nextState,self.depth,1)
            if value<cur_v:
                max_act=[act]
                value=cur_v
            elif value==cur_v:
                max_act.append(act)
        return max_act[random.randint(0,len(max_act)-1)]
                
        
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
        def Minimax_Value(self,gameState,depth,alpha,beta,agentIndex=0):
            if depth<=0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            actions=gameState.getLegalActions(agentIndex)
            if agentIndex==0:
                v=-1e5
                for action in actions:
                    nextState=gameState.generateChild(agentIndex,action)
                    v=max(v,Minimax_Value(self,nextState,depth-1,alpha,beta,agentIndex+1))
                    if v>=beta:return v
                    alpha=max(alpha,v)
                return v
            else:
                v=1e5
                for action in actions:
                    nextState=gameState.generateChild(agentIndex,action)
                    v=min(v,Minimax_Value(self,nextState,depth,alpha,beta,(agentIndex+1)%(gameState.getNumAgents())))
                    if v<=alpha:return v
                    beta=min(beta,v)
                return v
        
        max_act=gameState.getLegalActions(0)[random.randint(0,len(gameState.getLegalActions(0))-1)]     #设置一个默认动作
        value=-1e5
        a=-1e5
        b=1e5
        for act in gameState.getLegalActions(0):
            nextState=gameState.generateChild(0,act)
            cur_v=Minimax_Value(self,nextState,self.depth,a,b,1)
            if value<cur_v:
                max_act=[act]
                value=cur_v
            elif value==cur_v:
                max_act.append(act)
        return max_act[random.randint(0,len(max_act)-1)]
        util.raiseNotDefined()
#python pacman.py -p AlphaBetaAgent -l MyMaze -a depth=4