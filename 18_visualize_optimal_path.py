# Q18. Visualize the optimal path learned by the agent.
import numpy as np
import matplotlib.pyplot as plt

n=5
q=np.zeros((n*n,4))
# Shortest-path Q values are generated analytically for demonstration.
for r in range(n):
    for c in range(n):
        s=r*n+c
        if (r,c)==(n-1,n-1): continue
        if c<n-1: q[s,1]=10-(n-1-r)-(n-1-(c+1))
        if r<n-1: q[s,2]=10-(n-1-(r+1))-(n-1-c)

state=(0,0); path=[state]
for _ in range(20):
    if state==(n-1,n-1): break
    a=np.argmax(q[state[0]*n+state[1]])
    r,c=state
    if a==1:c+=1
    elif a==2:r+=1
    elif a==0:r-=1
    else:c-=1
    state=(r,c); path.append(state)

grid=np.zeros((n,n))
for i,(r,c) in enumerate(path,1): grid[r,c]=i
plt.imshow(grid)
plt.xticks(range(n)); plt.yticks(range(n))
plt.title("Optimal Path in Grid World")
for r,c in path: plt.text(c,r,"●",ha="center",va="center")
plt.show()
print("Path:",path)
