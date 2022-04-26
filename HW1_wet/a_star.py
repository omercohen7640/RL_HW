from puzzle import *
from planning_utils import *
import heapq
import datetime

def in_heapq(heapq_list, heapq_item_string):
    for i in  range(len(heapq_list)):
        if heapq_list[i][1] == heapq_item_string:
            return i
    return -1

def a_star(puzzle):
    '''
    apply a_star to a given puzzle
    :param puzzle: the puzzle to solve
    :return: a dictionary mapping state (as strings) to the action that should be taken (also a string)
    '''

    # general remark - to obtain hashable keys, instead of using State objects as keys, use state.as_string() since
    # these are immutable.

    initial = puzzle.start_state
    goal = puzzle.goal_state

    # the fringe is the queue to pop items from
    fringe = [(0, initial)]
    # concluded contains states that were already resolved
    concluded = set()
    # a mapping from state (as a string) to the currently minimal distance (int).
    distances = {initial.to_string(): 0}
    # the return value of the algorithm, a mapping from a state (as a string) to the state leading to it (NOT as string)
    # that achieves the minimal distance to the starting state of puzzle.
    prev = {initial.to_string(): None}

    while len(fringe) > 0:
        flag = False
        priority_s, current_s = heapq.heappop(fringe)  # get vertex u with minimal d + h
        concluded.add(current_s.to_string())  # add the vertex to S
        for action in current_s.get_actions():
            next_state = current_s.apply_action(action)
            if next_state.to_string() in concluded:
                continue
            idx_in_frienge = in_heapq(fringe, next_state.to_string())
            if idx_in_frienge != -1:
            #  if d[next_state] > d[u] + c(next_state,v):
                    if fringe[idx_in_frienge][0] >
            #       d[next_state] > d[u] + c(next_state,v)
            #       change priority in fringe
            #       update prev
            # else:
            #   create it, and add it to fringe
            #   update prev
            if next_state.to_string() not in concluded or distances[next_state.to_string()] > 1 + next_state.get_manhattan_distance(goal):
                heapq.heappush(fringe, (priority_s + 1 + next_state.get_manhattan_distance(goal), next_state))
                prev[next_state.to_string()] = action  # new
                if next_state.to_string() == goal.to_string():  # new
                    flag = True

        """flag = False
        for s in new_state:
            prev[s.to_string()] = current_s.to_string()
            if s.to_string() == goal.to_string():
                flag = True"""
        if flag:
            break

    return prev


def solve(puzzle):
    # compute mapping to previous using dijkstra
    prev_mapping = a_star(puzzle)
    # extract the state-action sequence
    plan = traverse(puzzle.goal_state, prev_mapping)
    print_plan(plan)
    return plan


if __name__ == '__main__':
    # we create some start and goal states. the number of actions between them is 25 although a shorter plan of
    # length 19 exists (make sure your plan is of the same length)
    initial_state = State()
    actions = [
        'r', 'r', 'd', 'l', 'u', 'l', 'd', 'd', 'r', 'r', 'u', 'l', 'd', 'r', 'u', 'u', 'l', 'd', 'l', 'd', 'r', 'r',
        'u', 'l', 'u'
    ]
    goal_state = initial_state
    for a in actions:
        goal_state = goal_state.apply_action(a)
    puzzle = Puzzle(initial_state, goal_state)
    print('original number of actions:{}'.format(len(actions)))
    solution_start_time = datetime.datetime.now()
    solve(puzzle)
    print('time to solve {}'.format(datetime.datetime.now()-solution_start_time))
