"""
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from game import Directions
from game import Agent
from game import Actions
import util
import time
import search


class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP


#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################


class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs
      aStarSearch or astar


    Note: You should NOT change any code in SearchAgent
    """
    def __init__(self,
                 fn='depthFirstSearch',
                 prob='PositionSearchProblem',
                 heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn +
                                 ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(
                    heuristic +
                    ' is not a function in searchAgents.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' %
                  (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(
                prob + ' is not a search problem type in SearchAgents.py.')
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None:
            raise Exception("No search function provided for SearchAgent")
        starttime = time.perf_counter_ns()
        problem = self.searchType(state)  # Makes a new search problem
        self.actions = self.searchFunction(problem)  # Find a path
        totalCost = problem.getCostOfActionSequence(self.actions)
        print('Path found with total cost of %d in %.f ns' %
              (totalCost, time.perf_counter_ns() - starttime))
        if '_expanded' in dir(problem):
            print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP


class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, child
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """
    def __init__(self,
                 gameState,
                 costFn=lambda x: 1,
                 goal=(1, 1),
                 start=None,
                 warn=True,
                 visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1
                     or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(
                        __main__._display):  #@UndefinedVariable
                    __main__._display.drawExpandedCells(
                        self._visitedlist)  #@UndefinedVariable

        return isGoal

    def expand(self, state):
        """
        Returns child states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (child, action, stepCost), where 'child' is a
         child to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that child
        """

        children = []
        for action in self.getActions(state):
            nextState = self.getNextState(state, action)
            cost = self.getActionCost(state, action, nextState)
            children.append((nextState, action, cost))

        # Bookkeeping for display purposes
        self._expanded += 1  # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return children

    def getActions(self, state):
        possible_directions = [
            Directions.NORTH, Directions.SOUTH, Directions.EAST,
            Directions.WEST
        ]
        valid_actions_from_state = []
        for action in possible_directions:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                valid_actions_from_state.append(action)
        return valid_actions_from_state

    def getActionCost(self, state, action, next_state):
        assert next_state == self.getNextState(
            state, action), ("Invalid next state passed to getActionCost().")
        return self.costFn(next_state)

    def getNextState(self, state, action):
        assert action in self.getActions(state), (
            "Invalid action passed to getActionCost().")
        x, y = state
        dx, dy = Actions.directionToVector(action)
        nextx, nexty = int(x + dx), int(y + dy)
        return (nextx, nexty)

    def getCostOfActionSequence(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x, y = self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x, y))
        return cost


def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ((xy1[0] - xy2[0])**2 + (xy1[1] - xy2[1])**2)**0.5


#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################


class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(),
                      startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0  # DO NOT CHANGE
        self.heuristicInfo = {
        }  # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def expand(self, state):
        "Returns child states, the actions they require, and a cost of 1."
        children = []
        self._expanded += 1  # DO NOT CHANGE
        for action in self.getActions(state):
            next_state = self.getNextState(state, action)
            action_cost = self.getActionCost(state, action, next_state)
            children.append((next_state, action, action_cost))
        return children

    def getActions(self, state):
        possible_directions = [
            Directions.NORTH, Directions.SOUTH, Directions.EAST,
            Directions.WEST
        ]
        valid_actions_from_state = []
        for action in possible_directions:
            x, y = state[0]
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                valid_actions_from_state.append(action)
        return valid_actions_from_state

    def getActionCost(self, state, action, next_state):
        assert next_state == self.getNextState(
            state, action), ("Invalid next state passed to getActionCost().")
        return 1

    def getNextState(self, state, action):
        assert action in self.getActions(state), (
            "Invalid action passed to getActionCost().")
        x, y = state[0]
        dx, dy = Actions.directionToVector(action)
        nextx, nexty = int(x + dx), int(y + dy)
        nextFood = state[1].copy()
        nextFood[nextx][nexty] = False
        return ((nextx, nexty), nextFood)

    def getCostOfActionSequence(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x, y = self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost


class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"

    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(
            prob, foodHeuristic)
        self.searchType = FoodSearchProblem


def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
    # nodes expanded:6862!!!!!
    # timecost: 1.605s!!!!
    # 又快又好!!!
    #
    # Honor code:室友胡（非软院，上个学期做过原版的project）提供的autograder等测试文件，以及关于Corner search+枚举方法的讨论。
    #              与 蒋哲宇、孙沛瑜、李松泽等同学讨论了多种启发函数，以及他们的合理性+可行性分析。
    #
    # 方法简述：
    # 从整个地图中"巧妙"地选取一个点作为坐标原点，建立一个平面直角坐标系，利用四个象限，维护一个四边形。
    # 四边形的顶点分别在四个象限中，也是我们要吃的四个食物。这相当于是原问题的一个relaxation问题。
    # （实际上，这个四边形会退化成三角形、线段、点，但只要针对每种退化情况也设计求解即可。）
    #
    # 求解这个问题需要的一个基本的前提引理：
    # 在曼哈顿距离中，两个食物点(A,B)之间的距离实际上是一个"矩形"，如果存在"一个"另外的点C在这个矩形中，
    # 这个点将会不需要任何额外的路径开销就可以被吃掉。并且，如果按(A,C,B)的顺序寻路，得到的距离将会与(A,B)相同。
    # 类似地，把这个性质称为曼哈顿距离的三角关系
    #
    # 另一个基本的引理:在划分好的平面直角坐标系中(x,y轴不做任何角度旋转，与格子对齐)
    # 在相邻的两个象限中分别选择两个点，它们的路径矩形一定不会进入另外两个象限。
    # 而在相对的两个象限中选择两个点，它们的路径矩形在四个象限中都会有
    #
    # 有了上面两个trivial的观点，分析一下我们面前的问题：分别在四个象限的四个食物，以及任意位置的一个Pacman，如何求解最短路径。
    # 先研究如何吃掉这四个食物（按分布的象限一、二、三、四，记作ABCD)，我们选择的策略如下：
    # 考察A-B-C-D-A这个闭环的长度,因为前面的第一个引理保证:
    # 如果我们走斜边(A,C)，当B或D其中一个点（不妨设为B）在(A,C)的矩形中时，我们研究(A,B,C)与研究(A,C)的结果一致。
    # 但如果B,D都不在这个矩形中，或者B,D都在这个矩形中（从一个对角到另一个对角，只能选择对角线的上半面或下半面，一定有一个象限是无法到达的）
    # 这时我们就会与实际路径长度出现明显的偏差（即使是在这个relaxation问题中）
    # 而如果按象限顺序，研究相邻象限之间的点之间的距离组合，这个矩形不会与另外的点产生任何交集，同时保证比前面的情况更优（更接近真实路径长度）
    #
    # 实际上，我们并不需要走一个闭环，而是这个闭环去掉一个边，再从被去掉的边的两端选择一个点与pacman的位置进行连边
    # 则我们枚举这个矩形的所有边（一共也就4条），计算:去掉每条边L时，pacman分别与L两端的点连接时的总最短路径。
    # 最笨的实现也只需要4*2=8种情况取最优
    # 这时，我们就得到了这个relaxtion问题的一个下界，同样也是原问题的一个下界。将这个下界作为启发值。
    #
    # 当有一个食物被吃掉，我们就损失了四边形的一个顶点，这时我们选择一个新的顶点，或者问题退化，并且保证良定义性质。将在实验报告中详细说明。
    #
    # 而为了进一步压缩拓展点数，我们需要让启发值尽可能地优，同时一定程度上让开始阶段的启发值更大。进行优化：
    # 在选择坐标原点时，尽可能保证四个象限中都有点，避免我们的问题在开始阶段就进行退化。
    # 对每个象限中的点，按照他们与原点的距离降序进行排序，优先选择离原点远的点构成四边形，这样构成的四个点更可能得到更长的路径
    # 决定路径后，对比原问题，如果还存在不在任意一个路径矩形中的点，则再适当放大启发值。
    # (如果通过按象限枚举所有点，求出一个最长路径，虽然一定得到更优的路径，但当一个点被吃掉后很难证明一致性，而且当前的选点方式已经足够拿到bonus)

    # 记录pacman，事物信息
    position, foodGrid = state
    foodList = foodGrid.asList()
    # 叉乘函数（本来用来验证三点共线，但是没用上）
    """
    def cross(vec1, vec2):
        return vec1[0] * vec2[1] - vec2[0] * vec1[1]
    """

    # 在搜索开始前，先选定坐标原点，并建立每个食物到四个象限的映射
    if state == problem.getStartState():
        # 选择食物坐标在x,y方向的最值
        if len(foodList) == 0:
            minx = maxx = miny = maxy = 0
        else:
            minx = min(food[0] for food in foodList)
            maxx = max(food[0] for food in foodList)
            miny = min(food[1] for food in foodList)
            maxy = max(food[1] for food in foodList)

        #先选择一个参考点，为了近似地选出在 左上、左下、右上、右下的四个点
        startpoint = ((maxx + minx) / 2, (maxy + miny) / 2)
        leftup = (0, 0)
        leftdown = (1e5, 1e5)
        rightup = (0, 0)
        rightdown = (0, 0)

        # 通过计算到参考点的x,y坐标组合，近似寻找在四个象限中离原点最远的点，即食物分布最外围的"四个角"
        for food in foodList:
            if food[0] + food[1] > rightup[0] + rightup[1]:
                rightup = food
            if food[0] - food[1] > rightdown[0] - rightdown[1] and food[
                    1] - startpoint[1] < 0:
                rightdown = food
            if -food[0] - food[1] > -leftdown[0] - leftdown[1]:
                leftdown = food
            if -food[0] + food[1] > -leftup[0] + leftup[1] and food[
                    0] - startpoint[0] < 0:
                leftup = food

        # 直线公式Ax+By+C=0, A=Y2-Y1 B=X1-X2 C=X2Y1-X1Y2
        # 连接 右上-左下，左上-右下 两条直线，计算他们的交点作为坐标原点
        # 可以尽可能保证四个象限中都有点
        A1 = rightup[1] - leftdown[1]
        B1 = leftdown[0] - rightup[0]
        C1 = rightup[0] * leftdown[1] - leftdown[0] * rightup[1]

        A2 = leftup[1] - rightdown[1]
        B2 = rightdown[0] - leftup[0]
        C2 = leftup[0] * rightdown[1] - rightdown[0] * leftup[1]

        # 点到直线的距离公式：|Ax+By+C|/sqrt{A^2+B^2}
        # 直线交点：A1x+B1y+C1=0,A2x+B2y+C2=0
        # 解得x=(C2B1-C1B2)/m y=(C1A2-C2A1)/m
        m = A1 * B2 - A2 * B1
        if m != 0:
            x, y = int((C2 * B1 - C1 * B2) / m), int(
                (C1 * A2 - C2 * A1) / m)  # 取整处理，因为需要计算食物到原点的曼哈顿距离
            if (x, y) not in foodList:  # 如果选定的点没有食物，就选这个点作为原点
                problem.heuristicInfo['origin'] = (x, y)
            else:  # 如果选定的点有食物，则尝试这个点周围的四个点
                if (x, y + 1) not in foodList:  # 尽可能不让原点有食物
                    problem.heuristicInfo['origin'] = (x, y + 1)
                elif (x + 1, y) not in foodList:
                    problem.heuristicInfo['origin'] = (x + 1, y)
                elif (x - 1, y) not in foodList:
                    problem.heuristicInfo['origin'] = (x - 1, y)
                elif (x, y - 1) not in foodList:
                    problem.heuristicInfo['origin'] = (x, y - 1)
                else:
                    problem.heuristicInfo['origin'] = (
                        x, y)  # 如果继续搜索，可能会导致过分偏移，导致某个象限空了
                    # 权衡两种损失，选择让原点有食物，影响不大
        else:
            problem.heuristicInfo['origin'] = (int(
                (maxx + minx) / 2), int(
                    (maxy + miny) / 2))  #直线交点不存在，选择一个近似的中心点作为原点

        # 建立食物到四个象限的映射。对坐标轴上的点，也归到某一象限中去。
        problem.heuristicInfo['1qua'] = []
        problem.heuristicInfo['2qua'] = []
        problem.heuristicInfo['3qua'] = []
        problem.heuristicInfo['4qua'] = []
        for food in foodList:
            x = food[0] - problem.heuristicInfo['origin'][0]
            y = food[1] - problem.heuristicInfo['origin'][1]
            if x == 0:
                if y > 0:
                    problem.heuristicInfo['1qua'].append(food)
                    problem.heuristicInfo[food] = 1
                else:
                    problem.heuristicInfo['3qua'].append(food)
                    problem.heuristicInfo[food] = 3
            elif y == 0:
                if x > 0:
                    problem.heuristicInfo['4qua'].append(food)
                    problem.heuristicInfo[food] = 4
                else:
                    problem.heuristicInfo['2qua'].append(food)
                    problem.heuristicInfo[food] = 2
            else:
                if x > 0 and y > 0:
                    problem.heuristicInfo['1qua'].append(food)
                    problem.heuristicInfo[food] = 1
                if x < 0 and y > 0:
                    problem.heuristicInfo['2qua'].append(food)
                    problem.heuristicInfo[food] = 2
                if x < 0 and y < 0:
                    problem.heuristicInfo['3qua'].append(food)
                    problem.heuristicInfo[food] = 3
                if x > 0 and y < 0:
                    problem.heuristicInfo['4qua'].append(food)
                    problem.heuristicInfo[food] = 4
        # 对每个象限中的食物，按照其到坐标原点的曼哈顿距离降序排序
        (problem.heuristicInfo['1qua']).sort(
            key=lambda s: -util.manhattanDistance(
                s, problem.heuristicInfo['origin']))
        (problem.heuristicInfo['2qua']).sort(
            key=lambda s: -util.manhattanDistance(
                s, problem.heuristicInfo['origin']))
        (problem.heuristicInfo['3qua']).sort(
            key=lambda s: -util.manhattanDistance(
                s, problem.heuristicInfo['origin']))
        (problem.heuristicInfo['4qua']).sort(
            key=lambda s: -util.manhattanDistance(
                s, problem.heuristicInfo['origin']))
    # 初始化部分完成

    # 特判剩余食物数量，在食物少的时候，可以更简单也更准确地表示其最短路径
    if not len(foodList):
        return 0
    if len(foodList) == 1:
        return util.manhattanDistance(position, foodList[0])
    if len(foodList) == 2:
        return min(util.manhattanDistance(position, foodList[0]),
                   util.manhattanDistance(position, foodList[1]))
        +util.manhattanDistance(foodList[0], foodList[1])
    # 食物数量大，利用我们设计的方法进行处理
    if len(foodList) > 2:
        quad = []  # 四边形四个顶点

        # 选择四边形的四个顶点
        # 这种选择方式，保证每次吃掉一个顶点，只会从当前象限中选择新节点，或退化。
        # 一定不影响其他的点。
        for node in problem.heuristicInfo['1qua']:
            if node in foodList:
                quad.append(node)
                break
        for node in problem.heuristicInfo['2qua']:
            if node in foodList:
                quad.append(node)
                break
        for node in problem.heuristicInfo['3qua']:
            if node in foodList:
                quad.append(node)
                break
        for node in problem.heuristicInfo['4qua']:
            if node in foodList:
                quad.append(node)
                break

        Ranges = []  # 两点之间的边，即路径矩形的范围

        # 只有一个象限有点 选距离pacman最远的点进行评估
        if len(quad) == 1:
            ans = 0
            extra = 0  # 附加值
            endnode = 0  # 端点
            node = 0
            for food in foodList:
                if util.manhattanDistance(position, food) > ans:
                    ans = util.manhattanDistance(position, food)
                    node = food
                Ranges.append(
                    (min(position[0], food[0]), max(position[0], food[0]),
                     min(position[1], food[1]), max(position[1], food[1])))
                endnode = food  # 记录下一个端点，另一个端点是position，即pacman位置
            # 如果存在额外的食物点不在任何一个路径矩形的范围内时，此时距离实际最短路径一定有偏差
            # 如果这个额外的食物点在我们选择的路径的两个端点的紧邻时，我们的当前估计与最短路径最少差1（吃掉这个额外点）
            # 如果不在，则我们的当前估计与最短路径最少差2：不管是从两端延伸出一条新边，还是修改边从而偏离我们目前的路径再返回，最少开销都是2

            for food in foodList:
                for Range in Ranges:
                    if food[0] >= Range[0] and food[0] <= Range[1] and food[
                            1] >= Range[2] and food[1] <= Range[3]:
                        extra = 0
                        break
                    else:
                        extra = 1
                if extra != 0:
                    if util.manhattanDistance(
                            food, endnode) == 1 or util.manhattanDistance(
                                food, position) == 1:
                        ans += 1
                    else:
                        ans += 2
                    break
            return ans
        # 只有两个象限有点：两个点连起来，pacman再选一个离自己最近的点
        elif len(quad) == 2:
            extra = 0  # 附加值
            endnode = 0  # 端点
            ans = util.manhattanDistance(quad[0], quad[1])
            Ranges.append(
                (min(quad[1][0], quad[0][0]), max(quad[1][0], quad[0][0]),
                 min(quad[1][1], quad[0][1]), max(quad[1][1], quad[0][1])))
            if util.manhattanDistance(position,
                                      quad[0]) > util.manhattanDistance(
                                          position, quad[1]):
                ans += util.manhattanDistance(position, quad[1])
                Ranges.append((min(position[0],
                                   quad[1][0]), max(position[0], quad[1][0]),
                               min(position[1],
                                   quad[1][1]), max(position[1], quad[1][1])))
                endnode = quad[0]
            else:
                ans += util.manhattanDistance(position, quad[0])
                Ranges.append((min(position[0],
                                   quad[0][0]), max(position[0], quad[0][0]),
                               min(position[1],
                                   quad[0][1]), max(position[1], quad[0][1])))
                endnode = quad[1]
            # 如果存在额外的食物点不在任何一个路径矩形的范围内时，此时距离实际最短路径一定有偏差
            # 如果这个额外的食物点在我们选择的路径的两个端点的紧邻时，我们的当前估计与最短路径最少差1（吃掉这个额外点）
            # 如果不在，则我们的当前估计与最短路径最少差2：不管是从两端延伸出一条新边，还是修改边从而偏离我们目前的路径再返回，最少开销都是2

            for food in foodList:
                for Range in Ranges:
                    if food[0] >= Range[0] and food[0] <= Range[1] and food[
                            1] >= Range[2] and food[1] <= Range[3]:
                        extra = 0
                        break
                    else:
                        extra = 1
                if extra != 0:
                    if util.manhattanDistance(
                            food, endnode) == 1 or util.manhattanDistance(
                                food, position) == 1:
                        ans += 1
                    else:
                        ans += 2
                    break
            return ans
        # 有三个或四个象限有点，此时已经可以连成闭环了，当成同种情况处理。
        else:
            ringpath = 0  #闭环长度
            ans = 1e5
            Ranges = []  #路径矩形集合
            for i in range(len(quad)):
                ringpath += util.manhattanDistance(quad[i],
                                                   quad[(i + 1) % len(quad)])
                Ranges.append((min(quad[(i + 1) % len(quad)][0], quad[i][0]),
                               max(quad[(i + 1) % len(quad)][0], quad[i][0]),
                               min(quad[(i + 1) % len(quad)][1], quad[i][1]),
                               max(quad[(i + 1) % len(quad)][1], quad[i][1])))
            deledge = 0  #被删除的边（路径矩形）
            poslink = 0  #pacman(position)决定连接的点
            endnode = 0  #端点
            minring = 1e5  #枚举后得到的最小路径
            for i in range(len(quad)):  #枚举每条边删去，并且让position选一个端点连接，选择出最小路径
                if util.manhattanDistance(
                        position, quad[i]) > util.manhattanDistance(
                            position, quad[(i + 1) % len(quad)]):
                    if ringpath - util.manhattanDistance(
                            quad[i], quad[
                                (i + 1) % len(quad)]) + util.manhattanDistance(
                                    position, quad[
                                        (i + 1) % len(quad)]) < minring:
                        deledge = (min(quad[(i + 1) % len(quad)][0],
                                       quad[i][0]),
                                   max(quad[(i + 1) % len(quad)][0],
                                       quad[i][0]),
                                   min(quad[(i + 1) % len(quad)][1],
                                       quad[i][1]),
                                   max(quad[(i + 1) % len(quad)][1],
                                       quad[i][1]))
                        poslink = (min(position[0],
                                       quad[(i + 1) % len(quad)][0]),
                                   max(position[0],
                                       quad[(i + 1) % len(quad)][0]),
                                   min(position[1],
                                       quad[(i + 1) % len(quad)][1]),
                                   max(position[1],
                                       quad[(i + 1) % len(quad)][1]))
                        endnode = quad[i]
                        minring = ringpath - util.manhattanDistance(
                            quad[i], quad[
                                (i + 1) % len(quad)]) + util.manhattanDistance(
                                    position, quad[(i + 1) % len(quad)])
                else:
                    if ringpath - util.manhattanDistance(
                            quad[i], quad[
                                (i + 1) % len(quad)]) + util.manhattanDistance(
                                    position, quad[i]) < minring:
                        deledge = (min(quad[(i + 1) % len(quad)][0],
                                       quad[i][0]),
                                   max(quad[(i + 1) % len(quad)][0],
                                       quad[i][0]),
                                   min(quad[(i + 1) % len(quad)][1],
                                       quad[i][1]),
                                   max(quad[(i + 1) % len(quad)][1],
                                       quad[i][1]))
                        poslink = (min(position[0], quad[i][0]),
                                   max(position[0], quad[i][0]),
                                   min(position[1], quad[i][1]),
                                   max(position[1], quad[i][1]))
                        endnode = quad[(i + 1) % len(quad)]
                        minring = ringpath - util.manhattanDistance(
                            quad[i], quad[(i + 1) %
                                          len(quad)]) + util.manhattanDistance(
                                              position, quad[i])
            Ranges.remove(deledge)  #删除该删除的边
            Ranges.append(poslink)  #添加新的连边
            ans = minring
            # 如果存在额外的食物点不在任何一个路径矩形的范围内时，此时距离实际最短路径一定有偏差
            # 如果这个额外的食物点在我们选择的路径的两个端点的紧邻时，我们的当前估计与最短路径最少差1（吃掉这个额外点）
            # 如果不在，则我们的当前估计与最短路径最少差2：不管是从两端延伸出一条新边，还是修改边从而偏离我们目前的路径再返回，最少开销都是2
            for food in foodList:
                add = 0.5
                for Range in Ranges:
                    if food[0] >= Range[0] and food[0] <= Range[1] and food[
                            1] >= Range[2] and food[1] <= Range[3]:
                        extra = 0
                        break
                    else:
                        extra = 0.5
                if extra != 0:
                    if util.manhattanDistance(
                            food, endnode) == 1 or util.manhattanDistance(
                                food, position) == 1:
                        ans += 1
                    else:
                        ans += 2
                    break
            return ans

# 实现方法到这里就结束了 后面是一些曾经尝试过并且没有被删去的方法。

# bfs方法预处理：在开始前搜索出图上每两个节点之间的实际最近距离，再存储起来。
# 是rollout方法！禁用！（但是效果真的很好，破百也有可能）
    """
    def bfs(food):              ##
        queue=util.Queue()
        exploredSet=util.Counter()
        froniterSet=util.Counter()
        queue.push((food,0))
        while 1:
            if queue.isEmpty():break
            (state,nowcost)=queue.pop()
            #print(state,nowcost)
            exploredSet[state[0]]=1
            problem.heuristicInfo[food[0],state[0]]=nowcost
            problem.heuristicInfo[state[0],food[0]]=nowcost
            actions=problem.getActions(state)
            expandList=[]
            for act in actions:
                expandList.append(problem.getNextState(state,act))
            for newstate in expandList:
                if exploredSet.__getitem__(newstate[0])==0 and froniterSet.__getitem__(newstate[0])==0:
                    queue.push((newstate,nowcost+1))
                    froniterSet[newstate]=1
    """

# 距离函数：本意用来对图上两点之间的曼哈顿距离进行一些小小的优化，但是好像没啥太大用。
    """
    def distance(pos1, i):
        step=2
        tdis = util.manhattanDistance(pos1, i)
        if (pos1[0] - i[0]) == 0 and (pos1[1] - i[1]) > 0:  #North
            x = i[0]
            y = i[1]
            while y < pos1[1]:
                if wallMap[(6 - y) * 21 + x] == 'T':
                    tdis += step
                    break
                y += 1
        if (pos1[0] - i[0]) == 0 and (pos1[1] - i[1]) < 0:  #South
            x = i[0]
            y = pos1[1]
            while y < i[1]:
                if wallMap[(6 - y) * 21 + x] == 'T':
                    #print(pos1)
                    #print(i)
                    #print(x,y)
                    tdis += step
                    break
                y += 1
        if (pos1[0] - i[0]) > 0 and (pos1[1] - i[1]) == 0:  #East
            x = i[0]
            y = i[1]
            while x < pos1[0]:
                if wallMap[(6 - y) * 21 + x] == 'T':
                    tdis += step
                    break
                x += 1
        if (pos1[0] - i[0]) < 0 and (pos1[1] - i[1]) == 0:  #West
            x = pos1[0]
            y = i[1]
            while x < i[0]:
                if wallMap[(6 - y) * 21 + x] == 'T':
                    tdis += step
                    break
                x += 1
        return tdis
    """

# corner search:每次取3-4个点，计算这几个点的路径的排列组合，找到一个最小值！
#               但是简单选取效果不好，如果枚举4个点，每次要做O(n^4)的操作，复杂度有点高，而且常数很大，实际应用起来很慢。
# related work:室友胡（非软院，非本课程）的实现方法大致如上，对corner search的结果进行了一些优化。expanded node:6800+ 但是运行时间很慢，在15-20s之间。
    """
    if len(foodList) == 0: return 0
    if len(foodList) == 1: return distance(position, foodList[0])
    if len(foodList) == 2:
        return min(
            distance(position, foodList[0]),
            distance(position, foodList[1])) + distance(
                foodList[0],
                foodList[1])  #util.manhattanDistance(foodList[0], foodList[1])
    if len(foodList) >= 3:
        #dis01 = util.manhattanDistance(foodList[0], foodList[1])
        #dis02 = util.manhattanDistance(foodList[0], foodList[2])
        #dis12 = util.manhattanDistance(foodList[1], foodList[2])
        dis01 = distance(foodList[0], foodList[1])
        dis02 = distance(foodList[0], foodList[2])
        dis12 = distance(foodList[1], foodList[2])
        dis0 = min(dis01, dis02)
        dis1 = min(dis01, dis12)
        dis2 = min(dis02, dis12)
        path0 = dis0 + dis12
        path1 = dis1 + dis02
        path2 = dis2 + dis01
        return min(
            distance(position, foodList[0]) + path0,
            distance(position, foodList[1]) + path1,
            distance(position, foodList[2]) + path2)
    else:
        #foodList.sort(key=lambda x: -util.manhattanDistance(x, position))
        #dis01 = util.manhattanDistance(foodList[0], foodList[1])
        #dis02 = util.manhattanDistance(foodList[0], foodList[2])
        #dis03 = util.manhattanDistance(foodList[0], foodList[3])
        #dis12 = util.manhattanDistance(foodList[1], foodList[2])
        #dis13 = util.manhattanDistance(foodList[1], foodList[3])
        #dis23 = util.manhattanDistance(foodList[2], foodList[3])
        foodList.sort(key=lambda x: distance(x, position))
        dis01 = distance(foodList[0], foodList[1])
        dis02 = distance(foodList[0], foodList[2])
        dis03 = distance(foodList[0], foodList[3])
        dis12 = distance(foodList[1], foodList[2])
        dis13 = distance(foodList[1], foodList[3])
        dis23 = distance(foodList[2], foodList[3])

        dis0 = min(dis01 + dis12 + dis23, dis01 + dis13 + dis23,
                   dis02 + dis12 + dis13, dis02 + dis23 + dis13,
                   dis03 + dis13 + dis12, dis03 + dis23 + dis12)
        dis1 = min(dis01 + dis02 + dis23, dis01 + dis03 + dis23,
                   dis12 + dis23 + dis03, dis12 + dis02 + dis03,
                   dis13 + dis03 + dis02, dis13 + dis23 + dis02)
        dis2 = min(dis02 + dis01 + dis13, dis02 + dis03 + dis13,
                   dis12 + dis13 + dis03, dis12 + dis01 + dis03,
                   dis23 + dis03 + dis01, dis23 + dis13 + dis01)
        dis3 = min(dis03 + dis01 + dis12, dis03 + dis02 + dis12,
                   dis13 + dis12 + dis02, dis13 + dis01 + dis02,
                   dis23 + dis02 + dis01, dis23 + dis12 + dis01)
        return min(
            distance(position, foodList[0]) + dis0,
            distance(position, foodList[1]) + dis1,
            distance(position, foodList[2]) + dis2,
            distance(position, foodList[3]) + dis3)
    """

# 两个节点的最远距离+离这两个点最近的距离 非良定义
    def func(foodList1):
        if len(foodList1) == 0: return 0
        if len(foodList1) == 1: return distance(position, foodList1[0])
        if len(foodList1) == 2:
            return min(distance(position, foodList1[0]),
                       distance(position, foodList1[1])) + distance(
                           foodList1[0], foodList1[1])
        if len(foodList1) > 2:
            foodList.sort(key=lambda x: distance(x, position))
            return distance(foodList[0], foodList[1]) + min(
                distance(position, foodList[0]), distance(
                    position, foodList[1]))
            """
            pair=[]
            maxdis=0
            for i in foodList1:
                for j in foodList1:
                    if i==j:continue
                    if distance(i,j)>maxdis:
                        maxdis=distance(i,j)
                        pair=[i,j]
            dis=min(distance(position,pair[0]),distance(position,pair[1]))
            return dis+maxdis
            """
# 最近距离+最近点到最远点距离 良定义：8287

    def func1(foodList1):
        ans = 1e5
        if len(foodList1) == 0: return 0
        if len(foodList1) == 1: return distance(position, foodList1[0])
        if len(foodList1) == 2:
            return min(distance(position, foodList1[0]),
                       distance(position, foodList1[1])) + distance(
                           foodList1[0], foodList1[1])
        if len(foodList1) > 2:
            dis1 = 1e5
            dis2 = 0
            nearest = 0
            for i in foodList1:
                if dis1 > distance(position, i):
                    nearest = i
                    dis1 = distance(position, i)
            for i in foodList1:
                if i != nearest:
                    if dis2 < distance(i, nearest):
                        dis2 = distance(i, nearest)
            return dis1 + dis2

    """
    import math
    position, foodGrid = state
    foodList=foodGrid.asList()
    """
    #foodList.reverse()
# x方向relaxation or y方向relaxtion 非良定义，但是稍微改进一下就可成为良定义，但效果不佳。 8k+
    """
    def func2(foodList1):
        ans=1e5
        if len(foodList1)==0:return 0
        if len(foodList1)==1:return distance(position,foodList1[0])
        if len(foodList1)==2:
            return min(distance(position,foodList1[0]),distance(position,foodList1[1]))+distance(foodList1[0],foodList1[1])
        if len(foodList1)>2:
            dis1=1e5
            foodList1x=sorted(foodList1,key=lambda s:s[0])
            foodList1y=sorted(foodList1,key=lambda s:s[1])
            target=0
            for i in range(len(foodList1x)-1):
                xsum=1e5
                disx1=0
                disx2=0
                for j in range(len(foodList1x)-1):
                        if foodList1x[j+1][0]<=foodList1x[i][0]:
                            disx1+=(foodList1x[j+1][0]-foodList1x[j][0])
                        if foodList1x[j][0]>=foodList1x[i][0]:
                            disx2+=(foodList1x[j+1][0]-foodList1x[j][0])
                if disx1>disx2: disx1,disx2=disx2,disx1
                if (disx1*2+disx2)<xsum:
                    target=i
                    xsum=disx1*2+disx2
            return xsum+abs(foodList1x[target][0]-position[0])
    return func2(foodList)
    """
# 按x划分子问题or 按y划分子问题 (本来想尝试的分治思想，但效果不佳）#非良定义
    """
    if len(foodList)==0:return 0
    if len(foodList)==1:return distance(position,foodList[0])
    if len(foodList)==2:
        return min(distance(position,foodList[0]),distance(position,foodList[1]))+distance(foodList[0],foodList[1])
    if len(foodList)>2:
        dis1=1e5
        foodListx=sorted(foodList,key=lambda s:s[0])
        foodListy=sorted(foodList,key=lambda s:s[1])
        for i in range(len(foodListx)):
            if foodListx[i][0]>position[0]:break
        xgap=abs(foodListx[i][0]-foodListx[i-1][0])#+min(position[0]-foodListx[i-1][0],foodListx[i][0]-position[0])
        sublist1=foodListx[:i]
        sublist2=foodListx[i:]
        ysolution=func2(sublist1)+func2(sublist2)#max(func(sublist1),func1(sublist1))+max(func(sublist2),func1(sublist2))

        for i in range(len(foodListy)):
            if foodListy[i][0]>position[0]:break
        ygap=abs(foodListy[i][1]-foodListy[i-1][1])#+min(position[1]-foodListy[i-1][1],foodListy[i][1]-position[1])
        sublist1=foodListy[:i]
        sublist2=foodListy[i:]
        xsolution=func2(sublist1)+func2(sublist2)#+max(func(sublist1),func1(sublist1))+max(func(sublist2),func1(sublist2))
        return min(xsolution,ysolution)#multi function 7483
    """
# 最近点+最远点 非良定义！
    """
    import math
    position, foodGrid = state
    foodList=foodGrid.asList()
    wallMap=problem.walls.__str__()
    ans=1e5
    if len(foodList)==0:return 0 
    if len(foodList)==1:
        tdis=util.manhattanDistance(position,foodList[0])
        i=foodList[0]
        if (position[0]-i[0])==0 and (position[1]-i[1])>0:        #North 
            x=i[0]
            y=i[1]
            while y<position[1]:
                if wallMap[(6-y)*21+x]=='T':
                    tdis+=2
                    break
                y+=1
        if (position[0]-i[0])==0 and (position[1]-i[1])<0:        #South
            x=i[0]
            y=position[1]
            while y<i[1]:
                if wallMap[(6-y)*21+x]=='T':
                    #print(position)
                    #print(i)
                    #print(x,y)
                    tdis+=2
                    break
                y+=1
        if (position[0]-i[0])>0 and (position[1]-i[1])==0:        #East
            x=i[0]
            y=i[1]
            while x<position[0]:
                if wallMap[(6-y)*21+x]=='T':
                    tdis+=2
                    break
                x+=1
        if (position[0]-i[0])<0 and (position[1]-i[1])==0:        #Wast
            x=position[0]
            y=i[1]
            while x<i[0]:
                if wallMap[(6-y)*21+x]=='T':
                    tdis+=2
                    break
                x+=1
        return tdis
    if len(foodList)==2:
        i=foodList[0]
        j=foodList[1]
        tdis=util.manhattanDistance(i,j)
        if (j[0]-i[0])==0 and (j[1]-i[1])>0:        #North 
            x=i[0]
            y=i[1]
            while y<j[1]:
                if wallMap[(6-y)*21+x]=='T':
                    tdis+=2
                    break
                y+=1
        if (j[0]-i[0])==0 and (j[1]-i[1])<0:        #South
            x=i[0]
            y=j[1]
            while y<i[1]:
                if wallMap[(6-y)*21+x]=='T':
                    tdis+=2
                    break
                y+=1
        if (j[0]-i[0])>0 and (j[1]-i[1])==0:        #East
            x=i[0]
            y=i[1]
            while x<j[0]:
                if wallMap[(6-y)*21+x]=='T':
                    tdis+=2
                    break
                x+=1
        if (j[0]-i[0])<0 and (j[1]-i[1])==0:        #Wast
            x=j[0]
            y=i[1]
            while x<i[0]:
                if wallMap[(6-y)*21+x]=='T':
                    tdis+=2
                    break
                x+=1
        return min(util.manhattanDistance(position,foodList[0]),util.manhattanDistance(position,foodList[1]))+tdis
    if len(foodList)>2:
        dis1=1e5
        dis2=0
        dis3=1e5
        for i in foodList:
            tdis=util.manhattanDistance(position,i)
            if (position[0]-i[0])==0 and (position[1]-i[1])>0:        #North 
                x=i[0]
                y=i[1]
                while y<position[1]:
                    if wallMap[(6-y)*21+x]=='T':
                        tdis+=2
                        break
                    y+=1
            if (position[0]-i[0])==0 and (position[1]-i[1])<0:        #South
                x=i[0]
                y=position[1]
                while y<i[1]:
                    if wallMap[(6-y)*21+x]=='T':
                        #print(position)
                        #print(i)
                        #print(x,y)
                        tdis+=2
                        break
                    y+=1
            if (position[0]-i[0])>0 and (position[1]-i[1])==0:        #East
                x=i[0]
                y=i[1]
                while x<position[0]:
                    if wallMap[(6-y)*21+x]=='T':
                        tdis+=2
                        break
                    x+=1
            if (position[0]-i[0])<0 and (position[1]-i[1])==0:        #Wast
                x=position[0]
                y=i[1]
                while x<i[0]:
                    if wallMap[(6-y)*21+x]=='T':
                        tdis+=2
                        break
                    x+=1
            dis1=min(dis1,tdis)
            for j in foodList:
                if i!=j:
                    tdis=util.manhattanDistance(i,j)
                    if (j[0]-i[0])==0 and (j[1]-i[1])>0:        #North 
                        x=i[0]
                        y=i[1]
                        while y<j[1]:
                            if wallMap[(6-y)*21+x]=='T':
                                tdis+=2
                                break
                            y+=1
                    if (j[0]-i[0])==0 and (j[1]-i[1])<0:        #South
                        x=i[0]
                        y=j[1]
                        while y<i[1]:
                            if wallMap[(6-y)*21+x]=='T':
                                tdis+=2
                                break
                            y+=1
                    if (j[0]-i[0])>0 and (j[1]-i[1])==0:        #East
                        x=i[0]
                        y=i[1]
                        while x<j[0]:
                            if wallMap[(6-y)*21+x]=='T':
                                tdis+=2
                                break
                            x+=1
                    if (j[0]-i[0])<0 and (j[1]-i[1])==0:        #Wast
                        x=j[0]
                        y=i[1]
                        while x<i[0]:
                            if wallMap[(6-y)*21+x]=='T':
                                tdis+=2
                                break
                            x+=1
                    dis2=max(dis2,tdis)
        return dis1+dis2
    """
# 每次锁定一个节点 非良定义
    """
    if problem.heuristicInfo.get('target')==None or problem.heuristicInfo['eat']==1:
        for xy in foodList:
            if ans>=abs(xy[0]-position[0])+abs(xy[1]-position[1]):
            #if ans<=math.sqrt((position[0]-xy[0])**2+(position[1]-xy[1])**2):
                ans=abs(xy[0]-position[0])+abs(xy[1]-position[1])
                #ans=math.sqrt((position[0]-xy[0])**2+(position[1]-xy[1])**2)
                problem.heuristicInfo['target']=xy
    else: ans=abs(problem.heuristicInfo['target'][0]-position[0])+abs(problem.heuristicInfo['target'][1]-position[1])
    #else: ans=math.sqrt((position[0]-problem.heuristicInfo['target'][0])**2+(position[1]-problem.heuristicInfo['target'][1])**2) 
    problem.heuristicInfo['eat']=0
    if ans==0:
        problem.heuristicInfo['eat']=1
    return ans
    ####
    #target on only one food 
    ####
    xy=foodList[-1]
    return (xy[0]-position[0])+abs(xy[1]-position[1])

    if len(foodList)==1:#12253
        xy=foodList[0]    
    #ans=min(ans,math.sqrt((position[0]-xy[0])**2+(position[1]-xy[1])**2))
        ans=abs(xy[0]-position[0])+abs(xy[1]-position[1])


    if len(foodList)>=2: #13297
        xy1=foodList[0]
        xy2=foodList[-1]
        ans=min(abs(xy1[0]-position[0])+abs(xy1[1]-position[1]),abs(xy2[0]-position[0])+abs(xy2[1]-position[1]))
    return ans
    """
    util.raiseNotDefined()