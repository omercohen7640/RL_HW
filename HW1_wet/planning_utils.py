def traverse(goal_state, prev):
    '''
    extract a plan using the result of dijkstra's algorithm
    :param goal_state: the end state
    :param prev: result of dijkstra's algorithm
    :return: a list of (state, actions) such that the first element is (start_state, a_0), and the last is
    (goal_state, None)
    '''
    result = [(goal_state, None)]
    # remove the following line and complete the algorithm
    while True:
        action = prev[goal_state.to_string()]
        if action == 'u':
            a = 'd'
        elif action == 'd':
            a = 'u'
        elif action == 'l':
            a = 'r'
        elif action == 'r':
            a = 'l'
        else:
            break
        goal_state = goal_state.apply_action(a)
        result.append((goal_state, action))



    return result


def print_plan(plan):
    print('plan length {}'.format(len(plan)-1))
    for current_state, action in plan:
        print(current_state.to_string())
        if action is not None:
            print('apply action {}'.format(action))
