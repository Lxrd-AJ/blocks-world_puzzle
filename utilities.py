import matplotlib.pyplot as plt

def format_time(seconds):
    if seconds < 400:
        s = float(seconds)
        return "%.1f seconds" % (s,)
    elif seconds < 4000:
        m = seconds / 60.0
        return "%.2f minutes" % (m,)
    else:
        h = seconds / 3600.0
        return "%.2f hours" % (h,)
    
def observe( state, size ):
    # TODO: Incomplete
    fig, ax = plt.subplots()
    s = state 
    s[state.index('A')] = 20
    s[state.index('B')] = 30
    s[state.index('C')] = 40
    if _AGENT_ in state:
        s[state.index(_AGENT_)] = 50
    maze = np.array(s).reshape((size,size))
    print(maze)
    
    # plt.matshow( maze )
    # plt.show()
    
    # ax.matshow(maze, cmap=plt.cm.Blues)

    # for i in range(size):
    #     for j in range(size):
    #         c = maze[j,i]
    #         ax.text(i, j, str(c), va='center', ha='center')