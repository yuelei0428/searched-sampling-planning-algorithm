import numpy as np
# import heapq
from pqdict import pqdict
from tqdm import tqdm

class Node:
  def __init__(self, parent, position, g, h, epsilon):
    self.parent = parent
    self.position = position
    self.h = h
    self.g = g

    self.f = g + epsilon * h
    self.id = tuple(position)

  
  def __eq__(self, other):
    return np.abs(self.position[0] - other.position[0]) <= 0.001 \
      and np.abs(self.position[1] - other.position[1]) <= 0.001 \
      and np.abs(self.position[2] - other.position[2]) <= 0.001

  def __lt__(self, other):
    return self.f < other.f

# def collision_check(start, end, blocks):
#   for j in range(len(blocks)):
#       origin = start
#       end = end
#       tmin = 0
#       tmax = np.linalg.norm(end - origin)
#       ray = (end - origin) / np.linalg.norm(end - origin)
#       flag = False
#       for k in range(3):
#         if ray[k] == 0:
#           flag = True
#           break
#         t0 = (blocks[j][k] - origin[k]) / (ray[k])
#         t1 = (blocks[j][k+3] - origin[k]) / (ray[k])
#         if ray[k] < 0:
#           t0, t1 = t1, t0
#         tmin = max(tmin, t0)
#         tmax = min(tmax, t1)
#         if tmax < tmin:
#           flag = True # flag = True when no collision for this block
#           break
#       if flag:
#         continue
#       else:
#         return True
#   return False


# return true if there is collision
# https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
def collision_check(start, end, blocks):
  for j in range(len(blocks)):
    if collision_check_single_block(start, end, blocks[j]):
      return True
  return False

def collision_check_single_block(start, end, block):
  dirfrac_x = 1.0 / (end[0] - start[0]) if (end[0] - start[0]) != 0 else 1e10
  dirfrac_y = 1.0 / (end[1] - start[1]) if (end[1] - start[1]) != 0 else 1e10
  dirfrac_z = 1.0 / (end[2] - start[2]) if (end[2] - start[2]) != 0 else 1e10

  t1 = (block[0] - start[0]) * dirfrac_x
  t2 = (block[3] - start[0]) * dirfrac_x
  t3 = (block[1] - start[1]) * dirfrac_y
  t4 = (block[4] - start[1]) * dirfrac_y
  t5 = (block[2] - start[2]) * dirfrac_z
  t6 = (block[5] - start[2]) * dirfrac_z

  tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6))
  tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6))

  if tmax < 0:
    return False
  if tmin > tmax:
    return False

  return True

class MyPlanner:
  __slots__ = ['boundary', 'blocks']
  
  def __init__(self, boundary, blocks):
    self.boundary = boundary
    self.blocks = blocks


  def plan(self,start,goal):
    numofdirs = 26
    stepsize = 0.05
    [dX,dY,dZ] = np.meshgrid([-1,0,1],[-1,0,1],[-1,0,1])
    dR = np.vstack((dX.flatten(),dY.flatten(),dZ.flatten()))
    # print('dR shape', dR.shape) # (3, 27)
    dR = np.delete(dR,13,axis=1)
    # dR = dR / np.sqrt(np.sum(dR**2,axis=0))
    dR = dR * stepsize
    
    # weighted A* search
    # define the heuristic function h_i = \|x_\tau - x_i\|_2
    
    closed = dict()
    epsilon = 1 # epsilon >= 1
    path = []
    start_node = Node(parent=None, position=[start[0], start[1], start[2]], g=0, h=np.linalg.norm(start - goal), epsilon=epsilon)
    open = pqdict()
    open.additem(start_node.id, start_node)

    count = 0
    while len(open) > 0:
      id, node = open.popitem()
      closed[node.id] = node
      count += 1

      if np.linalg.norm(node.position - goal) < stepsize:
        while node is not None:
          path.append(node.position)
          node = node.parent
        break

      for k in range(numofdirs):
        next_position = node.position + dR[:,k]
        next_position = np.round(next_position, 3)

        if( next_position[0] <= self.boundary[0,0] or next_position[0] >= self.boundary[0,3] or \
            next_position[1] <= self.boundary[0,1] or next_position[1] >= self.boundary[0,4] or \
            next_position[2] <= self.boundary[0,2] or next_position[2] >= self.boundary[0,5] ):
          continue
        
        collision = collision_check(node.position, next_position, self.blocks)
        
        if collision:
          continue
        
        next_g = node.g + np.linalg.norm(dR[:,k])
        next = Node(node, next_position, next_g, np.linalg.norm(next_position - goal), epsilon)

        # Check if next is in closed
        if next.id in closed:
          continue

        # Check if next is in open
        if next.id in open:
          if open[next.id].g > next.g:
            open[next.id] = next
          # elif open[next.id].g == next.g and open[next.id].h > next.h:
          #   open[next.id] = next
        else:
          open.additem(next.id, next)

    print("path", path)
    return np.array(path[::-1])
    
    '''''''''''
    for _ in range(2000):
      mindisttogoal = 1000000
      node = None
      for k in range(numofdirs):
        next = path[-1] + dR[:,k]
        
        # Check if this direction is valid
        if( next[0] < self.boundary[0,0] or next[0] > self.boundary[0,3] or \
            next[1] < self.boundary[0,1] or next[1] > self.boundary[0,4] or \
            next[2] < self.boundary[0,2] or next[2] > self.boundary[0,5] ):
          continue
        
        valid = True
        for k in range(self.blocks.shape[0]):
          if( next[0] >= self.blocks[k,0] and next[0] <= self.blocks[k,3] and\
              next[1] >= self.blocks[k,1] and next[1] <= self.blocks[k,4] and\
              next[2] >= self.blocks[k,2] and next[2] <= self.blocks[k,5] ):
            valid = False
            break
        if not valid:
          continue
        
        # Update next node
        disttogoal = sum((next - goal)**2)
        if( disttogoal < mindisttogoal):
          mindisttogoal = disttogoal
          node = next
      
      if node is None:
        break
      
      path.append(node)

      
      
      # Check if done
      if sum((path[-1]-goal)**2) <= 0.1:
        break
    '''''''''


    

