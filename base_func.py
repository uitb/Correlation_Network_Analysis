# -*- coding: UTF-8 -*-
import mdtraj as md
import numpy as np
import networkx as nx
import sys
import pickle
import random
import copy
import time
import string
import scipy as sp
import matplotlib as plt
from math import exp,log,sqrt
if 'ipykernel' in sys.modules:              #if the module is running in notebook
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


class Network(object):
    '''
    read trajs form traj files and first calculate residue correlation, then construct graph that all nodes correspond to residues. analysis communities and paths
    -----------------------------------------
    Attributes:
    traj: { mdtraj.core.trajectory.Trajectory }
    residue_num: { int }
                the residues num.
    traj_length: { int }
                traj length or frames num.
    correlation: { numpy.ndarray }
                matrix of correlation for residue pairs.
    protein_graph: { networkx.classes.graph.Graph }
                the protein network graph.
    communities: { list }
                all possible communities division
    modularity_q: { list }
                modularity Q value for all communities division.
    centrality: { list }
                centrality value for per node.
    paths: { dict }
                N suboptmal paths for all pairs of source and sink nodes, include shortest path, in all child list, the first element is path length and after it is the path nodes.
    hotpots: { dict }
                hotpots for all pairs of source and sink nodes
    path_num: [ list ]
                paths num for all pairs of source and sink nodes

    -----------------------------------------
    Functions:
    correlation_analysis(traj,top): calculate correlations between all residues;
    read_correlate(correlate_file,nodes_num): read correlation matrix for file;
    construct_graph(traj,top): using networkx.classes.graph.Graph to construct graph;
    community_analysis(G): analysisi communities by Girvan–Newman algorithm;
    calc_centrality(G): calculate all nodes centrality;
    calc_shortest_path(G): find all the shortest paths for any two nodes;
    suboptmal_paths_1(G,source,sink,desire_N): find desire_N number of suboptmal paths for source node and sink node; 
    '''

    def __init__(self,traj_files,top,stride = 1,lazy_load = False,traj_start=None,traj_end=None,
        contact_distance = 10.0,contact_ratio=0.75,correlation_threshold=0.6,
        paths_source=None,paths_sink=None,paths_num=500):
        '''
        traj_files: { str or list }
                    a traj file name or lists of multiple file's name.
        pdb: { str or mdtraj.core.trajectory.Trajectory }
                    pdb file path or matraj topfile.
        stride: { int } optional, default = 1
                    stride for read traj
        lazy_load: { bool } optional, default = False
                    whether to read traj more memory efficient than loading an entire trajectory at once.
        traj_start and traj_end: { int } optional,default = None
                    traj start frame and end frame to read, default read all frames.
        contact_distance: { float } optional, default = 10.0
                    the distance of residue pairs to generate edge, default is 10 angstrom.
        contact_ratio: { float } optional, default = 0.75
                    the ratio that residue pairs distance shorter than contact_distance in whole traj to generate edge.
        correlation_threshold: { float } optional, default = 0.6
                    the corelation value to generate edge.
        paths_source and paths_sink: { int } optional, default = None
                    the start and end node in graph to find paths, default is None, not find any paths. if not provide this, you could call suboptmal_paths_1 mathod to find paths and generate self.paths attribute
        paths_num: { int } optional, default = 500
                    How many paths needed.
        '''
        if isinstance(traj_files,str):
            self.traj_files = [traj_files]
        elif isinstance(traj_files,list):
            self.traj_files = traj_files
        else:
            print 'traj_fies should be  str or list type'
            raise TypeError
        if isinstance(top,str):
            self.top = md.load(top)
        elif isinstance(top,md.core.trajectory.Trajectory):
            self.top = top
        else:
            print 'top should be  str or md.core.trajectory.Trajectory type'
            raise TypeError
        self._stride = stride
        self._lazy_load = lazy_load
        self._start = traj_start
        self._end = traj_end
        self._contact_distance = contact_distance
        self._contact_ratio = contact_ratio
        self._correlation_threshold = correlation_threshold
        self._load_traj(self.traj_files)                                               #self.traj
        self.residue_num = self.traj.n_residues
        self.traj_length = self.traj.n_frames
        self.correlation_analysis(self.traj,self.top)                                    #self.correlation
        self.construct_graph(self.traj,self.top)                                         #self.protein_graph
        self.community_analysis(self.protein_graph) 
        self.calc_centrality(self.protein_graph)                        #self.communities and self.modularity_q ,self.centrality
        if paths_source and paths_sink:
            self.suboptmal_paths_1(self.protein_graph,paths_source,paths_sink,paths_num)   #self.paths

    def _load_traj(self,traj_files):
        '''
        read trajectory from a traj file or a lists of traj files
        '''
        if not self._lazy_load:
            self.traj = md.load(self.traj_files[0], top=self.top,stride = self._stride)[self._start:self._end]
            for t in self.traj_files[1:]:
                self.traj += md.load(t, top=self.top,stride = self._stride)[self._start:self._end]
        else:
            iter_traj = md.iterload(self.traj_files[0],top=self.top,chunk=500,stride=self._stride)
            self.traj = iter_traj.next()
            for chunk in iter_traj:
                self.traj += chunk
            self.traj = self.traj[self._start:self._end]
            for t in traj_files[1:]:
                iter_traj = md.iterload(t,top=self.top,chunk=500,stride=self._stride)
                traj_ = iter_traj.next()
                for chunk in iter_traj:
                    traj_ += chunk
                self.traj += traj_[self._start:self._end]
        self.traj.superpose(reference=self.top)

    def correlation_analysis(self,traj,top):
        '''
        analysis the correlation between any pair of residual from the trajctory by covariance matrix, generate self.graph attribute, and the values are between -1 and 1
        '''
        total_frames = len(traj)
        atoms = top.topology.select('name CA')
        xyz = [np.array(traj.xyz[:,i]) for i in atoms]
        self.residue_num = traj.n_residues
        self.traj_length = traj.n_frames
        self.correlation = np.zeros((self.residue_num, self.residue_num))
        
        mean = np.mean(xyz,axis=1)
        delta = np.array([xyz[i]-mean[i] for i in range(self.residue_num)])
        mean_dot = np.array([np.mean(np.array([np.dot(d[m],d[m]) for m in range(total_frames)])) for d in delta ])
        mean_dot_sqrt = np.sqrt(mean_dot)
        total = reduce(lambda x,y:x + y,range(self.residue_num)) + self.residue_num
        
        for i in tqdm(range(self.residue_num),desc='{}'.format(self.residue_num),file=sys.stdout,position=0):
            for j in tqdm(range(i,self.residue_num),desc='{}'.format(self.residue_num-i),leave=False,file=sys.stdout,position=1):
                CovIJ = np.mean(np.array([np.dot(delta[i,m],delta[j,m]) for m in range(total_frames)]))
                self.correlation[i,j] = CovIJ/(mean_dot_sqrt[i]*mean_dot_sqrt[j])
                self.correlation[j,i] = self.correlation[i,j]   
        sys.stdout.flush()

    def _find_edges(self,traj,top):
        '''
        calculate the ratio of residual pairs' distance shorter than threshold in whole trajctory and generate edeges

        '''
        atoms = top.topology.select("name CA")
        nodes_range = self.residue_num
        edges = np.zeros((nodes_range,nodes_range))
        #edges[edges==0] = (threshold+1)
        
        total_frames = len(traj)
        for current, frame in enumerate(tqdm(traj,desc=str(total_frames))):
            xyz = frame.xyz[0][atoms]
            dis = self._EuclideanDistances(xyz,xyz)*10
            dis[dis<self._contact_distance] = 1
            dis[dis>self._contact_distance] = 0
            edges += dis
        edges = edges/total_frames
        sys.stdout.flush()
        weighted_edges = []
        for i in tqdm(range(nodes_range),desc='{}'.format(nodes_range),file=sys.stdout,position=0):
            for j in tqdm(range(i+1,nodes_range),desc='{}'.format(nodes_range-i),leave=False,file=sys.stdout,position=1):
                if edges[i,j] >= self._contact_ratio and abs(float(self.correlation[i,j])) >= self._correlation_threshold:
                    wight = -log(abs(float(self.correlation[i,j]))) 
                    weighted_edges.append((i,j,wight))
        sys.stdout.flush()
        return weighted_edges

    def _EuclideanDistances(self,A, B):
        '''
        calculate any pairs element's distance in two matrix 
        -----------------------------------------
        A,B: matrix, A.shape = B.shape
        -----------------------------------------
        return numpy.array
        '''
        BT = B.transpose()
        # vecProd = A * BT
        vecProd = np.dot(A,BT)
        # print(vecProd)
        SqA =  A**2
        # print(SqA)
        sumSqA = np.matrix(np.sum(SqA, axis=1))
        sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
        # print(sumSqAEx)

        SqB = B**2
        sumSqB = np.sum(SqB, axis=1)
        sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))    
        SqED = sumSqBEx + sumSqAEx - 2*vecProd
        SqED[SqED<0]=0.0   
        ED = np.sqrt(SqED)
        return ED

    def read_correlate(self,correlate_file,nodes_num):
        '''read correlation from a stored txt file, this not used current.
        -----------------------------------------
        correlate_file: { str }
                        correlate file path and name
        nodes_num: { int }
                        num of residue
        -----------------------------------------
        return numpy.array, a correlation matrix
        '''
        correlate_txt = ''
        correlate = []
        with open(correlate_file,'r') as f:
            c = iter(f.read().split(' ')[:-1])
        for i in range(nodes_num):
            corre = []
            for j in range(nodes_num):
                corre.append(c.next())
            correlate.append(corre)
        correlate = np.array(correlate)
        return correlate

    def construct_graph(self,traj,top):
        '''
        constuc network graph and generate self.graph attribute
        '''  
        self.residue_num = traj.n_residues
        self.traj_length = traj.n_frames       
        weighted_edges = self._find_edges(traj,top)
        nodes = range(0,self.residue_num)

        self.protein_graph = nx.Graph()
        self.protein_graph.add_nodes_from(nodes)
        self.protein_graph.add_weighted_edges_from(weighted_edges)   

    def community_analysis(self,G):
        '''
        community analysis by Girvan–Newman algorithm, generate self.community and self.modularity_q attributes,which present protein communities and correspond modularity q value 
        '''

        comp = nx.algorithms.community.girvan_newman(G)
        self.communities = []
        self.modularity_q = []
        for c in comp:
            self.communities.append(c)
            self.modularity_q.append(nx.algorithms.community.quality.modularity(G,c))

    def calc_centrality(self,G):
        '''
        calculate nodes centrality, node centralities that assess the density of connections per node.
        reference: 
        '''
        nodes = G.nodes
        num_nodes = len(nodes)
        A = np.zeros((num_nodes,num_nodes))
        for i,x in enumerate(nodes):
            for j,y in enumerate(nodes):
                node_data = G.get_edge_data(x,y)
                if node_data:
                    weight = node_data['weight']
                    A[i][j] = exp(-weight)
        # self.centrality = np.ones(num_nodes)
        # while True:
        #     centra_ = np.ones(num_nodes)
        #     for i,x in enumerate(self.centrality):
        #         c = 0
        #         for j,y in enumerate(self.centrality):
        #             c += A[i][j] * y
        #         centra_[i] = c
        #     centra_ = centra_/np.max(centra_)
        #     dis = np.linalg.norm(self.centrality-centra_)
        #     sim = 1.0/(1.0+dis)
        #     self.centrality = centra_
        # #     print sim
        #     if sim>=1-1e-10:
        # #         print centrality
        # #         print centra_
        #         break
        # or
        eigenvalues,eigenvectors = np.linalg.eig(A)
        self.centrality = eigenvectors[:,np.where(eigenvalues==max(eigenvalues))[0][0]]
        self.centrality = abs(self.centrality)/max(abs(self.centrality))

    def calc_shortest_path(self,G):
        '''
        find the any pair node's shortest path for all pairs node in graph
        ''' 
        num_nodes = len(G.nodes)
        nodes_axis = range(1, num_nodes + 1)    
        self.shortest_path_dict = nx.all_pairs_dijkstra_path(G)

    def suboptmal_paths_1(self,G,source,sink,desire_N=500):
        '''
        find suboptmal paths for a pair of node
        -----------------------------------------
        source and sink: { int }
                        the start node and end node for find paths
        desire_N: { int } optional, default=500
                        how many paths needed.
        '''
        paths_ = []
        simple_paths = nx.algorithms.simple_paths.shortest_simple_paths(G = G,source=source,target=sink,weight='weight')
        for i in range(desire_N):
            path = simple_paths.next()
            length = self._get_path_length(path)
            self.paths_.append([length] + path)
        hotpots = self._hotpots(paths_)

        try:
            self.paths[(source,sink)] = paths_
        except:
            self.paths = dict()
            self.paths[(source,sink)] = paths_
        try:
            self.paths_num.append(desire_N)
        except:
             self.paths_num = [desire_N]
        try:
            self.hotpots[(source,sink)] = hotpots
        except:
            self.hotpots = dict()
            self.hotpots[(source,sink)] = hotpots


    @classmethod
    def _get_path_length(cls,path):
        '''
        calculate path's length
        -----------------------------------------
        path: { list or array-like }
        '''
        length = 0
        for i in range(len(path)-1):
            length += self.protein_graph.get_edge_data(path[i],path[i+1]).get('weight',0)
        return length

    @classmethod
    def _suboptmal_paths(cls,source,sink,shortest_path,desire_N=500,max_iter = 1000):
        '''
        Deprecated !!! not used now and replaced by suboptmal_paths_1 mathod.
        '''
        cutoff = self._get_path_length(shortest_path)
        lenth = len(shortest_path)
        if desire_N==1:
            return shortest_path
        
        path = [source]
        self.paths = []
        iter_ = 0
        n = desire_N *2
        while len(self.paths) < n and iter_ < max_iter:
            iter_ +=1
            cutoff = cutoff + 0.56 #float(desire_N*cutoff)/(100000)
            try:
                self.find_paths(self.paths,path,cutoff,sink,n,lenth)
            except Exception as e:
                print e
                break

        for i in range(len(self.paths)):
            length = get_length(self.paths[i],G)
            self.paths[i].insert(0,length)
        self.paths.sort()

    @classmethod
    def _find_paths(cls,paths,path,cutoff,sink,N,lenth,i=0,done_node=[]):
        '''Deprecated !!!'''
        i+=1
        done = copy.deepcopy(done_node)
        if len(paths) == N:
            sys.stdout.write(' '*50)
            raise FoundPaths(N)
        if path[-1] == sink:
            if path not in paths:
                paths.append(path)
        elif get_length(path,self.protein_graph)>cutoff:# or len(path)>(lenth*1.5):
            pass
        else:
            node_neighbors = [n for n in self.protein_graph.neighbors(path[-1])]
            random.shuffle(node_neighbors)
            done.append(path[-1])
            for n in node_neighbors:
                if n not in done:
                    p = copy.deepcopy(path)
                    p.append(n)
                    if get_length(p,self.protein_graph) <= cutoff and len(p) < lenth+5:
                        paths_ = find_paths(paths,p,cutoff,sink,N,lenth,i,done)

    class _FoundPaths(Exception):
        '''Deprecated !!!'''
        def __init__(self,N):
            err = 'Found {} paths'.format(N)
            Exception.__init__(self, err)

    @classmethod
    def _hotpots(self,paths):
        '''find hotpots in suboptmal pahts,hotpots mean that all suboptmal paths will pass them '''
        hot_nodes = paths[0][1:]
        for path in paths:
            hot_nodes = np.intersect1d(hot_nodes,path[1:])
        hots = [str(self.top.top.residue(h)) for h in hot_nodes]
        return hots

class Visualization(object):
    '''
    Visualization for communities and paths in protein network.
    -----------------------------------------
    Functions:
    path()s: generate a pymol script (pdb_name_paths.pml) for show paths in pymol;
    community(community_partition): generate a pymol script (pdb_name_community.pml) for show communities in pymol;
    plot_depiction_communities(community_partition,ax,radius,xr,yr,zr,R): depiction communities by different color and size circles.
    '''
    def __init__(self,network,pdb_filename,colors = None):
        '''
        network: { Network }
                the protein network instance.
        pdb_filename: { str }
                pdb file path and name.
        colors: { list } optional, default = None
                colors for different communities
        '''
        self.network = network
        self.pdb = pdb_filename
        self.pdb_name = ''
        if self.pdb.rfind('/') != -1:
            self.pdb_name = self.pdb[self.pdb.rfind('/')+1:-4]
        else:
            self.pdb_name = self.pdb[:-4]
        if not colors:
            colors = get_cmap('viridis').colors+get_cmap('inferno').colors+get_cmap('plasma').colors+get_cmap('magma').colors
            self.colors = [colors[i*(len(colors)/30)] for i in range(30)]
        else:
            self.colors = colors
        pass

    def paths(self):
        '''
        generate a pymol script (pdb_name_paths.pml) for show paths in pymol, need to open pymol by oneself
        '''
        subtimal_paths = self.network.paths
        pdb_top = self.network.top

        with open('{}_paths.pml'.format(self.pdb_name),'w') as f:
            for i,c in enumerate(self.colors):
                f.write('set_color color{}={}\n'.format(i,c))
            f.write('load {}\n'.format(self.pdb))
            for m,paths in enumerate(subtimal_paths):
                lines = set()
                first = set()
                f.write('################################### SHORTEST PATH ######################################\n')
                for n,p in enumerate(paths):
                    for i in range(1,len(p)-1):
                        res1 = pdb_top.top.residue(p[i]).index+1
                        res2 = pdb_top.top.residue(p[i+1]).index+1
                        lines.add('{}///{}/CA,{}///{}/CA'.format(self.pdb_name,res1,self.pdb_name,res2))
                        if n==0:
                            first.add('{}///{}/CA,{}///{}/CA'.format(self.pdb_name,res1,self.pdb_name,res2))
                            f.write('distance {}_p{}_{}_{},{}///{}/CA,{}///{}/CA\n'.format(self.pdb_name,m,0,i,self.pdb_name,res1,self.pdb_name,res2))
                            f.write('color color{}, {}_p{}_{}_{}\n'.format(m,self.pdb_name,m,0,i) )
                f.write('################################### SHORTEST PATH ######################################\n\n')
                lines = lines.difference(first)
                for i,l in enumerate(lines):
                    f.write('distance {}_p{}_{}_{},{}\n'.format(self.pdb_name,m,1,i,l))
                    f.write('color color{}, {}_p{}_{}_{}\n'.format(m,self.pdb_name,m,1,i) )
                f.write('\n')
            f.write('set dash_gap, 0\n')
            f.write('set dash_radius,0.25\n')
            f.write('set dash_radius,0.6,{}_p*_{}_*\n'.format(self.pdb_name,0,))
            f.write('hide labels,All\n')
            f.write('bg_color white\n')
            f.write('set cartoon_transparency, 0.6\n')
            f.write('disable all\n')

    def community(self,community_partition):
        '''
        generate a pymol script (pdb_name_community.pml) for show communities in pymol, need to open pymol by oneself
        -----------------------------------------
        community_partition: { int }
                    which number of iterations in Girvan–Newman algorithms, could select by index of the max modularity Q value
        '''
        community = self.network.community[community_partition]
        pdb_name = ''

        with open(self.pdb_name+'_community.pml','w') as f:
            for i,c in enumerate(self.colors):
                f.write('set_color color{}={}\n'.format(i,c))
            f.write('load {}\n'.format(self.pdb))
            for i in range(len(community)):
                f.write('color color{}, {} and (resid {})\n'.format(i,self.pdb_name,','.join([str(j+1) for j in community[i]])))
            f.write('bg_color white')

    def _community_correlation(self,community_partition):
        '''
        calculate two communities' correlation
        -----------------------------------------
        community_partition: { int }
                    which number of iterations in Girvan–Newman algorithms, could select by index of the max modularity Q value
        '''
        community = self.network.community[community_partition] 
        c_correlation = np.zeros((len(community),len(community)))
        for i in range(len(community)-1):
            for j in range(i,len(community)):
                corr = sum([self.network.protein_graph.get_edge_data(m,n)['weight'] if self.network.protein_graph.get_edge_data(m,n) else 0
                            for m in community[i] for n in community[j]])
                c_correlation[i][j] = corr
        return c_correlation
    def _x_rotal(self,M,angle):
        '''
        rotation coordinate in x axis
        '''
        R = np.array([[cos(-angle),0,sin(-angle)],
                 [0,1,0],
                 [-sin(-angle),0,cos(-angle)]])
        return np.dot(M,R)
    def _y_rotal(self,M,angle):
        '''same with _x_total'''
        R = np.array([[1,0,0],
                     [0,cos(-angle),-sin(-angle)],
                     [0,sin(-angle),cos(-angle)]])
        return np.dot(M,R)
    def _z_rotal(self,M,angle):
        '''same with _x_total'''
        R = np.array([[cos(-angle),-sin(-angle),0],
                 [sin(-angle),cos(-angle),0],
                 [0,0,1]])
        return np.dot(M,R)

    def plot_depiction_communities(self,community_partition,ax=None,radius=1.0,xr=0,yr=0,zr=0,R=10.0):
        '''
        depiction communities by different color and size circles, one could use radius, xr, yr, zr, R parameters to adjust the angle of view
        -----------------------------------------
        community_partition: { int }
                    which number of iterations in Girvan–Newman algorithms, could select by index of the max modularity Q value
        ax: { matplotlib.axes._subplots.AxesSubplot } optional,default = None
        radius: { float } optional, default = 1.0
                    the base size of circles.
        xr,yr,zr: { float } optional, default = 0
                    the angle for rotation in x y z axis, in radian unit
        R: { float } optional,default = 10.0
                    the distance of view

        '''
        community_xy,community_r = self._depiction_communities_position(community_partition,radius,xr,yr,zr,R)
        if ax == None:
            fig = plt.figure(figsize=(6,6))
            ax = plt.gca()
        community_correlate = self._community_correlation(community_partition)
        for i in range(len(community)-1):
            for j in range(i,len(community)):
                corr = abs(community_correlate[i][j])
                line = [community_xy[i], community_xy[j]]
                (line_xs, line_ys) = zip(*line)
                #ax.add_line(Line2D(line_xs, line_ys, linewidth=corr, color='gray',alpha=0.5))
                ax.plot(line_xs, line_ys,lw=corr,color='gray',alpha=0.5)
        for i,r in enumerate(community_r):
            ax.add_patch(Circle(xy=community_xy[i],radius=r,color=self.colors[i],zorder=i+10))

        plt.axis('scaled')
        plt.axis('equal') 

    def _depiction_communities_position(self,community_partition,radius=1,xr=0,yr=0,zr=0,R=10):
        '''
        calculate the position and size for all circles
        -----------------------------------------
        community_partition: { int }
                    which number of iterations in Girvan–Newman algorithms, could select by index of the max modularity Q value
        radius: { float } optional, default = 1.0
                    the base size of circles.
        xr,yr,zr: { float } optional, default = 0
                    the angle for rotation in x y z axis, in radian unit
        R: { float } optional,default = 10.0
                    the distance of view
        '''
        top = self.network.top
        ca_index = top.topology.select('name CA')
        community = self.network.community[community_partition]
        #ca_index.sort()
        xyz = top.xyz[0]
        community_xy = []
        community_r = []
        R = -R
        for i,c in enumerate(community):
            xyz_ = np.mean(np.array([xyz[ca_index[j]] for j in c]),axis=0)
            xyz_ = self._x_rotal(xyz_,xr)
            xyz_ = self._y_rotal(xyz_,yr)
            xyz_ = self._z_rotal(xyz_,zr)
            
            xy = [xyz_[0]*10.0/R,xyz_[1]*10.0/R]
            r = len(c)/479.0*radius
            community_xy.append(xy)
            community_r.append(r)
           # cir.append(Circle(xy = xy,radius=r,color = colors[i]))
        return community_xy,community_r

def DataLoad(file_name):
    '''
    load data from saved file
    -----------------------------------------
    file_name: { str }
                file path and name to load data.
    '''
    try:
        with open(file_name,'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        data = None
    return data

def DataSave(file_name,data):
    '''
    save data
    -----------------------------------------
    file_name: { str }
                file path and name to store data.
    data: { any type }
    '''
    try:
        with open(file_name,'wb') as f:
            pickle.dump(data,f)
    except Exception as e:
        print('saving data to file {} failure because of {}'.format(file_name,e))

def find_the_nearest_res_to_centroid(xyz_list):
    '''
    find the nearest residue to all coordinate geometric center.
    -----------------------------------------
    xyz_list: { array-like }
                a list of residue coordinate.
    '''
    xyz_list = np.array(xyz_list)
    if len(xyz_list) == 1:
        return 0
    
    centroid = np.mean(xyz_list,axis=0)
    distance = 99999
    resid = 0
    for i,xyz in enumerate(tqdm(xyz_list)):
        d = sqrt((xyz[0]-centroid[0])**2 + (xyz[1]-centroid[1])**2 +(xyz[2]-centroid[2])**2)
        if d < distance:
            resid = i
            distance = d
    return resid

def find_the_center_res_for_community(communities,pdb):
    '''
    find the nearest residue to community geometric center.
    -----------------------------------------
    communities: { list }
    pdb: { str }
                pdb file path and name.
    '''
    pdb_top = md.load(pdb)
    pdb_xyz = pdb_top.xyz[0]
    ca_index = pdb_top.topology.select('name CA')
    ca_index.sort()
    res = []
    for community in communities:
        comm = list(community)
        comm.sort()
        xyz = [pdb_xyz[ca_index[c]] for c in comm]
        res.append(comm[find_the_nearest_res_to_centroid(xyz)])
    return res

def plot_correlation(correlation,ax=None,figsize=(8,6),c_map = None):
    '''
    plot correlation
    -----------------------------------------
    correlation: { array-like }
                the correlation matrix
    ax: { matplotlib.axes._subplots.AxesSubplot } optional, default = None
    figsize: { tuple } optional,default = (8,6)
                the size for fig, only need for ax == None.
    c_map: { matplotlib.colors.LinearSegmentedColormap }
    '''
    if not ax:
        fig=plt.figure(figsize=figsize)
        ax = gca()
        show = True
    else:
        show = False
    if cmap == None:
        colors = [('white')] + [(plt.cm.jet(i)) for i in xrange(40,250)]
        c_map = plt.colors.LinearSegmentedColormap.from_list('new_map', colors, N=300)

    heatmap = ax.pcolor(correlation, cmap=c_map, vmin=-1, vmax=1)
    cbar = plt.pyplot.colorbar(heatmap, orientation="vertical",ax=ax)
    plt.pyplot.xticks(range(0,len(correlation),100)+[len(correlation)])
    plt.pyplot.yticks(range(0,len(correlation),100)+[len(correlation)])
    plt.pyplot.xlabel('Residue Index', fontsize=14)
    plt.pyplot.ylabel("Residue Index", fontsize=14)
    if show:
        fig.show()
