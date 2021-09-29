import pylab as plt
import numpy as np
import networkx as nx
### import pylab as plt


points_list = [(0, 1), (1, 5), (5, 6), (5, 4), (1, 2), (2, 3), (2, 7)]

goal = 7
mapping = {0: 'Start', 1: '1', 2: '2', 3: '3',
           4: '4', 5: '5', 6: '6', 7: '7-Destination'}
Graph = nx.Graph()
Graph.add_edges_from(points_list)
pos = nx.spring_layout(Graph, k=.5, center=points_list[2])
nx.draw_networkx_nodes(Graph, pos, node_color='g')
nx.draw_networkx_edges(Graph, pos, edge_color='b')
nx.draw_networkx_labels(Graph, pos)
plt.show()


numPoints = 8


rMatrix = np.matrix(np.ones(shape=(numPoints, numPoints)))
rMatrix *= -1

for point in points_list:
    print(point)
    if point[1] == goal:
        rMatrix[point] = 150
    else:
        rMatrix[point] = 0

    if point[0] == goal:
        rMatrix[point[::-1]] = 150
    else:
        rMatrix[point[::-1]] = 0


rMatrix[goal, goal] = 150
rMatrix


QMatrix = np.matrix(np.zeros([numPoints, numPoints]))

gamma = 0.8

initial_state = 1


def available_actions(state):
    current_state_row = rMatrix[state, ]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act


available_act = available_actions(initial_state)


def sample_next_action(available_actions_range):
    next_action = int(np.random.choice(available_act, 1))
    return next_action


action = sample_next_action(available_act)


def update(current_state, action, gamma):
    max_index = (np.where(QMatrix[action, ] == np.max(QMatrix[action, ]))[1])
    if (max_index.shape[0] > 1):
        max_index = int(np.random.choice(max_index, size=1))
    else:
        max_index = int(max_index)

    maxValue = QMatrix[action, max_index]
    QMatrix[current_state, action] = rMatrix[current_state, action] + gamma * maxValue
    print('maxValue', rMatrix[current_state, action] + gamma * maxValue)

    if (np.max(QMatrix) > 0):
        return(np.sum(QMatrix/np.max(QMatrix)*100))
    else:
        return (0)


update(initial_state, action, gamma)


scores = []

for i in range(700):
    currentState = np.random.randint(0, int(QMatrix.shape[0]))
    available_act = available_actions(currentState)
    action = sample_next_action(available_act)
    score = update(currentState, action, gamma)
    scores.append(score)
    print('Score:', str(score))

    print("Trained Q matrix:")
    print(QMatrix/np.max(QMatrix)*100)

currentState = 0
steps = [currentState]

while currentState != 7:

    next_step_index = np.where(
        QMatrix[currentState, ] == np.max(QMatrix[currentState, ]))[1]

    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size=1))
    else:
        next_step_index = int(next_step_index)

    steps.append(next_step_index)
    current_state = next_step_index


print("Most efficient path:")
print(steps)

plt.plot(scores)
plt.show()