import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import KDTree

class Filtration: 
    def __init__(self,points): 
        self.points = np.array(points)


    #Define 4 functions within Filtration which will match the different filtration types 

 
    def Height(self,direction): 
        direction_norm = direction/np.linalg.norm(direction)

        return self.points @ direction_norm



    def Radial(self, center):
        return np.linalg.norm(self.points - center, axis=1)

    def Density(self, k =5): 
        tree = KDTree(self.points)
        dists, _ = tree.query(self.points, k=k+1)

        return dists[:, -1]

    def Dilation(self):
        return squareform(pdist(self.points))


