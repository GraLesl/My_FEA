import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
import time
from scipy.sparse.linalg import spsolve
from scipy.linalg import solve_banded

# Stiffness Matrix Manipulation
# --------------------------------------------------------
def points_of_element(points, simplex):
    x = np.zeros(3)
    y = np.zeros(3)
    GPI = np.zeros(6) # Global Point indexes for each point in the simplex
    
    for i,elem in enumerate(simplex):
        x[i] = points[simplex[1-i]][0]
        y[i] = points[simplex[1-i]][1]
        
        GPI[2*i] = 2*elem
        GPI[2*i +1] = 2*elem + 1
    GPI = np.concatenate((GPI[-2:],GPI[0:-2]),axis=0)
    x = x[::-1]
    y = y[::-1]
    return x,y,GPI

def tri_area(points, simplex):
    #Area of triangle given vertecies given a simplex
    x, y, GPI = points_of_element(points, simplex)
    ones = np.ones(3)
    # build 3x3 matrix with rows [x_i, y_i, 1]
    matrix = np.vstack((x, y, ones)).T  # shape (3,3)
    area = 0.5 * abs(np.linalg.det(matrix))
    return area, x, y, GPI

def beta_matrix(points,simplex):

    A,x,y,GPI = tri_area(points,simplex)
    b1 = y[1] - y[2]
    b2 = y[2] - y[0]
    b3 = y[0] - y[1]
    g1 = x[2] - x[1]
    g2 = x[0] - x[2]
    g3 = x[1] - x[0]

    B = np.array([[b1,0,b2,0,b3,0],[0,g1,0,g2,0,g3],[g1,b1,g2,b2,g3,b3]]) * (1/(2*A))

    return B,x,y,GPI

def mat_matrix(Mat):
    # Plane Stress
    E = Mat.E
    v = Mat.v
    D = np.array([[1,v,0],[v,1,0],[0,0,(1-v)/2]]) * E / (1-v**2)
    # Plane Strain
    #D = ((E*(1-v))/((1+v)*(1-2*v))) * np.array([[1,v/(1-v),0],[v/(1-v),1,0],[0,0,(1-2*v)/(2*v)/(2*(1-v))]])
    return D

def localStiff(points,simplices,simp_num,D,t):
    A,x,y,GPI = tri_area(points,simplices[simp_num])
    B,x,y,GPI = beta_matrix(points,simplices[simp_num])
    Bt = np.linalg.matrix_transpose(B)

    kl = t * A * Bt @ D @ B

    return kl, GPI

def globalStiff(points,simplices,Mat):
    num_nodes = len(points)
    KG = lil_matrix((num_nodes*2,num_nodes*2))
    D = mat_matrix(Mat)

    for elem,simp in enumerate(simplices):
        
        kl, GPIl = localStiff(points, simplices, elem, D, Mat.t)
        GPIl = GPIl.astype(int)
        KG[np.ix_(GPIl, GPIl)] += kl
        '''
        for i in range(kl.shape[0]):
            for j in range(kl.shape[1]):
                
                globRow = int(GPIl[i])
                globCol = int(GPIl[j])
                KG[globRow,globCol] += kl[i,j]
        '''
    return KG

def constrain_via_free(gStiff, forces, fixed_points):
    """
    Robustly return reduced stiffness and force vectors using the "free DOFs" approach.
    fixed_points: 1D array/list of node indices (both DOFs fixed).
    """
    DOF = gStiff.shape[0]
    print(DOF)
    remove_idx = np.array([], dtype=int)
    if len(fixed_points):
        fixed_points = np.asarray(fixed_points, dtype=int)
        remove_idx = np.hstack([2*fixed_points, 2*fixed_points + 1])
        remove_idx = np.sort(remove_idx)

    all_dofs = np.arange(DOF, dtype=int)
    free = np.setdiff1d(all_dofs, remove_idx)

    KG_reduced = gStiff[np.ix_(free, free)]
    f_reduced = forces[free]

    return KG_reduced, f_reduced, free

def constrain_slider(gStiff, forces, fixed_points):
    DOF = gStiff.shape[0]

def create_forces(size,points, direction, totalForce):
    #creates a distributed load in one direction across a set of points
    forces = np.zeros(size)
    unitForce = totalForce / len(points)

    nudge = 0 if direction == 'x' else 1
    dof_indices = 2*points + nudge
    forces[dof_indices] += unitForce
    return forces
    
def add_forces(forces,points, direction, totalForce):
    #given a previously created force vector, add a new distributed load
    unitForce = totalForce / len(points)
    nudge = 0

    if direction == 'x':
        nudge = 0
    elif direction == 'y':
        nudge = 1
    else: 
        print('Error: incorrect dimension')

    for point in points:
        forces[point*2 + nudge] += unitForce
    
    return forces

def apply_deformation(points, displacements):

    new_points = points
    for i,point in enumerate(new_points):
            new_points[i,0] += displacements[0]
            new_points[i,1] += displacements[1]

            displacements = np.delete(displacements,(0,1),axis=0)
    
    

    return new_points

# 2D Mesh Generation
# --------------------------------------------------------
def mesh_rectangle(l,h,n_elm_l,n_elm_h):
    nodes = []

    for x in np.linspace(0,l,n_elm_l):
        for y in np.linspace(0,h,n_elm_h):
            nodes.append([x,y])

    points = np.array(nodes)

    tri = Delaunay(points)
    simplices = tri.simplices
    num_elem = len(tri.points)
    DOF = num_elem * 2

    return points,simplices,num_elem, DOF

def plot_mesh(points,simplices,figNum):
    points = points
    plt.figure(figNum)
    plt.triplot(points[:,0], points[:,1], simplices)
    plt.plot(points[:,0], points[:,1], 'o')

def mesh_semicircle(ri,ro,n_elem,n_angles):
    nodes = []

    for theta in np.linspace(0,3.14,num=n_angles):
        for r in np.linspace(ri,ro,num=n_elem):
            x = np.float64(np.cos(theta) * r)
            y = np.float64(np.sin(theta) * r)
            nodes.append([x,y])

    points = np.array(nodes)

    tri = Delaunay(points)
    simplices = tri.simplices
    num_elem = len(tri.points)
    DOF = num_elem * 2
    
    # Cut off extra simplices that generate inside the arc
    p = []
    r_cut = ri - 0.01
    for theta in np.linspace(0,3.14, num = n_angles):
        x = np.cos(theta) * r_cut
        y = np.sin(theta) * r_cut
        p.append([x,y])

    strings = tri.find_simplex(p)
    simplices = np.delete(simplices, strings[1:-1],0)

    return points,simplices,num_elem, DOF

# Boundary Conditions
# Put all boundary conditions including fixed DOFs and forces in one place
# Then use the apply method to constrain the global stiffness matrix
# -----------------------------------------------------------------------------
class BoundaryConditions:

    def __init__(self,ndof):
        self.ndof = ndof
        self.fixed_dofs = set()
        self.f = np.zeros(ndof)

    def fix_nodes(self, nodes, directions="xy"):

        for node in nodes:
            if "x" in directions:
                self.fixed_dofs.add(2*node)
            if "y" in directions:
                self.fixed_dofs.add(2*node+1)

    def add_point_force(self,node,fx,fy):
        self.f[2*node]   += fx
        self.f[2*node+1] += fy

    def add_distributed_load(self,nodes,fx,fy):
        unit_Load_X = fx / len(nodes)
        unit_Load_Y = fy / len(nodes)

        self.f[2*nodes]     += unit_Load_X
        self.f[2*nodes + 1] += unit_Load_Y

    def selectzone(self,rangeX, rangeY):
        self.rangeX = rangeX
        self.rangeY = rangeY

    def apply(self,K):

        all_dofs = np.arange(self.ndof)
        free_points = np.setdiff1d(all_dofs, list(self.fixed_dofs))

        K_reduced = K[np.ix_(free_points,free_points)]
        f_reduced = self.f[free_points]

        return K_reduced, f_reduced, free_points

class singleCondition:
    # Store all important information about a given condition (Force, DistLoad, DOFConst)
    def __init__(self,type,rangeX,rangeY,directions='xy',fx=0,fy=0):
        self.type = type
        self.rangeX = rangeX
        self.rangeY = rangeY
        self.directions = directions
        self.fx = fx
        self.fy = fy
        pass

# Material Class
class Material:
    def __init__(self,E,v,t):
        self.E = E
        self.v = v
        self.t = t

#Solve Function
# Given a global stiffness matrix and BC, solve for displacement
def solve_KG_BC(KG, BC):
    
    DOF = KG.shape[0]

    K_reduced, f_reduced, free_points = BC.apply(KG)
    print("Force", max(BC.f))
    print(type(K_reduced))
    # Solve for Displacement on all DOFS
    u_reduced = spsolve(K_reduced, f_reduced)
    u_reduced = u_reduced.flatten()

    # Apply displacements to all free DOFs leaving fixed ones locked
    u_full = np.zeros(DOF)
    u_full[free_points] = u_reduced

    maxDeformation = max(np.absolute(u_reduced))

    return u_full, maxDeformation

# Make Convergence Test
def convergence(geometry,Mat,Conditions,minNodes = 500, maxNodes = 5000,nData = 10):
    # Geometry is an array where the first index is its shape, and subsequent
    # indexes are the necessary parameters (MAYBE THIS SHOULD BE A CLASS)
    # Rectangular ['r',w,h]
    # SemiCircle ['sc', rm, width]
    # Mat props class containing E,v,t of a material
    # Boundary Conditions, a list of classes containing zones and their associated Boundary conditions
    #-------------
    # Objective is to incremenet up the number of nodes from the min to max
    # Make a plot of peak Deformation across nodes
    # Get a time readout for each mesh aswell

    num_Nodes_vector = np.linspace(minNodes,maxNodes,nData,dtype=int)
    num_Nodes_actual = []
    maxDisp = []
    time_to_solve = []
    for numNodes in num_Nodes_vector:
        start_time = time.perf_counter()
        
        # Create points and simlices based on geometry and current nodes
        geomType = geometry[0]
        if geomType == 'r':
            w = geometry[1]
            h = geometry[2]
            ratio = np.sqrt(w/h)

            persideN = round(np.sqrt(numNodes))
            nodesOnWidth = int(persideN * ratio)
            nodesOnHeight = int(persideN / ratio)

            nodes,simplices,num_nodes,DOF = mesh_rectangle(w,h,nodesOnWidth,nodesOnHeight)

        elif geomType =='sc':
            rm = geometry[1]
            width = geometry[2]
            ri = rm - 0.5*width
            ro = rm + 0.5*width

            ratio = np.sqrt(11) # ratio of number of elem/angle to number of angles

            persideN = round(np.sqrt(numNodes))
            nodesOnArc = int(persideN * ratio)
            nodesOnRow = int(persideN / ratio)

            nodes,simplices,num_nodes,DOF = mesh_semicircle(ri,ro,nodesOnRow,nodesOnArc)
        else:
            print('Incorect geometry type')
            break
        # Create a stiffness matrix
        KG = globalStiff(nodes,simplices,Mat)
        # Combine all boundary conditions
        Global_BC = BoundaryConditions(DOF)
        for i,condition in enumerate(Conditions):
            print("Condition Marker")
            # Add Each conditions parameters to the global boundary condition class
            hasDistLoad = False
            distLoadpoints = []
            print(condition.type)
            for i,node in enumerate(nodes):
                rangeX = condition.rangeX
                rangeY = condition.rangeY

                if min(rangeX) <= node[0] <= max(rangeX) and min(rangeY) <= node[1] <= max(rangeY):
                    print(condition.type)
                    match condition.type:
                        case 'Point_Load':
                            Global_BC.add_point_force([i],condition.fx,condition.fy)
                        case 'Distributed_Load':
                            # This one is weird
                            print("Got Distributed Load")
                            hasDistLoad = True
                            distLoadpoints.append(i)
                        case 'DOF_Constraint':
                            Global_BC.fix_nodes([i],condition.directions)
            
            if hasDistLoad:
                distLoadpoints = np.array([distLoadpoints]).flatten()
                Global_BC.add_distributed_load(distLoadpoints,condition.fx,condition.fy)
                
        
        
        # Solve for displacement
        u_full, maxDeformation = solve_KG_BC(KG,Global_BC)
        # Save Results (numNodes actual, maxDisp, time elpsed)   
        num_Nodes_actual.append(num_nodes)
        maxDisp.append(maxDeformation)
        
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        time_to_solve.append(elapsed_time)

    return num_Nodes_actual, maxDisp, time_to_solve

# Mesh Quality Check
def check_skew(points,simplecies):
    
    skewness = []

    for simplex in simplecies:
        x,y,GPI = points_of_element(points,simplex)

        p1 = np.array([x[0], y[0]])
        p2 = np.array([x[1], y[1]])
        p3 = np.array([x[2], y[2]])

        # Find Edge Lengths
        L12 = np.linalg.norm(p2 - p1)
        L23 = np.linalg.norm(p3 - p2)
        L31 = np.linalg.norm(p1 - p3)

        shortest = min(L12, L23,L31)
        perim = L12+L23+L31
        ratio = shortest / perim

        # Ideal ratio for a perfect element is 0.333
        # Multiply ratio by 3 to get a 0-1 value for skewness

        skewness.append(ratio * 3)
    
    avgSkew = np.average(skewness)
    minSkew = min(skewness)

    return avgSkew, minSkew

