import matplotlib.pyplot as plt
import matplotlib
import numpy as np 

    
class Grid():
    def __init__(self, height, width, costgrid):
        self.height = height
        self.width = width
        self.costgrid = costgrid 
        #print(costgrid)
        
        #draws the game grid on game initialization 
    def drawGrid(self):
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        ax.axis('off')
        
        #set backgrounf
        my_cmap = matplotlib.colors.ListedColormap(['w'])

        
        #draw vertical and horizontal lines 
        for x in range(self.width + 1): ax.axvline(x, lw=2, color='k', zorder=5)
        for y in range(self.height + 1): ax.axhline(y, lw=2, color='k', zorder=5)
        
        # draw the boxes
        ax.imshow(self.costgrid, interpolation='none', cmap=my_cmap, extent=[0, self.width, 0, self.height], zorder=0)
        
        for (row, col), z in np.ndenumerate(self.costgrid):
            #print("row: {}, col: {}, z: {}".format(row, col,z))
            ax.text(col+0.5,(self.height)-(row+0.5), '{:0.0f}'.format(z), ha='center', va='center', size = 'large')
        
        plt.show()
        
      #draws game grid, highlights the path passed   
    def drawPath(self, path):
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        ax.axis('off')
        
        #print("path: ", path)
        #temp = self.costgrid.copy()
        #path = [[0,0]]
        temp = np.ones(self.costgrid.shape) * np.nan
        for point in path:
            temp[point[0]][point[1]] = 1
        
        #set background and path color 
        my_cmap = matplotlib.colors.ListedColormap(['y'])
        my_cmap.set_bad(color='b', alpha=0)
        
        #draw vertical and horizontal lines 
        for x in range(self.width + 1): ax.axvline(x, lw=2, color='k', zorder=5)
        for y in range(self.height + 1): ax.axhline(y, lw=2, color='k', zorder=5)
        
        
        # draw the boxes
        ax.imshow(temp, interpolation='none', cmap=my_cmap, extent=[0, self.width, 0, self.height], zorder=0)
        
        for (row, col), z in np.ndenumerate(self.costgrid):
            #print("row: {}, col: {}, z: {}".format(row, col,z))
            ax.text(col+0.5,(self.height)-(row+0.5), '{:0.0f}'.format(z), ha='center', va='center', size = 'large')
        
        plt.show()