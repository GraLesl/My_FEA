import numpy as np
import FEATools as FEAtools
import matplotlib.pyplot as plt
import time
from scipy.sparse.linalg import spsolve


# Create a plot comparing number of elements to Maximum Deformation
#Test Parameters

nodes = np.array([500,1000,2000])
widths = np.array([100,150,200,500,800])
w = 1000
h = 500
t = 5
E = 210000
v = 0.3

material = FEAtools.Material(E,v,t)
Force = 800000
FigNum = 1
maxes = []
strains = []
deformations = []
Stresses = []

sigma = Force / (h * t) # Sigma in MPA
strain = sigma/E
Displacement = w * strain # in mm
print('Displacement :' + str(Displacement))
print('Strain :' + str(strain))
print('Stress :' + str(sigma))
# Function per mesh
# Create Mesh
def maxDefTest (width, height, n_elem, material, Force,fignum):

    start_time = time.perf_counter()

    ratio = np.sqrt(w/h)
    persideN = round(np.sqrt(n_elem))
    nodesOnWidth = int(persideN * ratio)
    nodesOnHeight = int(persideN / ratio)
    nodes,simplices,num_nodes,DOF = FEAtools.mesh_rectangle(width,height,nodesOnWidth,nodesOnHeight)
    KG = FEAtools.globalStiff(nodes,simplices,material)

    FEAtools.plot_mesh(nodes,simplices,fignum)
    nodeLocked = []
    nodeforced = []
    tol = 0.001
    for i,node in enumerate(nodes):
        if node[0] < tol:
            nodeLocked.append(i)
        
        if node[0] > width - tol:
            nodeforced.append(i)
    

    nodeLocked = np.array([nodeLocked]).flatten()
    nodeforced = np.array([nodeforced]).flatten()
    

    f = FEAtools.create_forces(DOF,nodeforced,'y',Force)


    KG_constrained, f_reduced, free = FEAtools.constrain_via_free(KG, f, nodeLocked)
    # safe solve:
    u_reduced = spsolve(KG_constrained, f_reduced)
    #u_reduced = np.linalg.solve(KG_constrained, f_reduced)
    u_reduced = u_reduced.flatten()
    # map back to full DOF vector if needed:
    u_full = np.zeros(DOF)
    u_full[free] = u_reduced

    maxDeformation = max(np.absolute(u_full))

    new_nodes = FEAtools.apply_deformation(nodes,u_full)
    
    plt.plot(new_nodes[:,0], new_nodes[:,1], 'o',)
    plt.title(str(n_elem))

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print("--- Diagnostics ---")
    print("Nodes:", n_elem, f"Execution time: {elapsed_time:.4f} seconds")
    print("KG shape:", KG.shape)
    print("f shape:", f.shape, "sum f:", np.sum(f))
    print("f/nodes:", np.sum(f)/nodesOnHeight)
    avgSkew, minSkew = FEAtools.check_skew(nodes,simplices)
    print("Average Skew:", avgSkew, "Min Skew", minSkew)


    return maxDeformation


for i, num in enumerate(nodes):
    deform = maxDefTest(w,h,num,material,Force,FigNum)
    testStrain = deform / w
    testStress = E * testStrain
    strains.append(deform / w)
    deformations.append(deform)
    Stresses.append(testStress)
    FigNum += 1


fig, ax1 = plt.subplots()

ax1.set_xlabel('Number of Nodes')
ax1.set_ylabel('Strain & Deformation')
l1 = ax1.plot(nodes,strains,color='r')
l2 = ax1.plot(nodes,deformations,color='b')

ax2 = ax1.twinx()

ax2.set_ylabel('Stress (MPA)')
l3 = ax2.plot(nodes,Stresses,color='g')

fig.tight_layout()
fig.legend(['Strains','Deformations','Stresses'],loc='upper left')
plt.show()
'''
plt.figure(FigNum)
plt.plot(nodes,strains)
plt.plot(nodes,Stresses)
plt.plot(nodes,deformations)
plt.legend(["Strains","Stresses","Deformations"])
plt.show()
'''
