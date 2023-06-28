import matplotlib.pyplot as plt
import operator
import networkx as nx
import numpy as np
import itertools
import datetime
import scipy as sp
from scipy.spatial import cKDTree
import random


# function to transform color image to grayscale
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def rgb2hex(color):
    '''
    Matplotlib scatter is not happy with rgb tuples so we need to transform them to hex
    '''
    c = tuple([np.int(255 if c == 1.0 else c * 256.0) for c in color])
    return "#%02x%02x%02x" % c

def get_edge_color(G):
    nodecolors = nx.get_node_attributes(G, 'color')
    x = []
    for eij in G.edges():
        if nodecolors[eij[0]]!=nodecolors[eij[1]]:
            if 0.5 < np.random.rand():
                x.append(nodecolors[eij[0]])
            else:
                x.append(nodecolors[eij[1]])
        else:
            x.append(nodecolors[eij[0]])

    return x

def get_edge_degree_probvec(G, percenti=25):
    x = []
    for eij in G.edges():
        x.append(G.degree(eij[0])+G.degree(eij[1]))
    
    x = np.array(x)
    cutoff = np.percentile(x, percenti)
    x[x < cutoff] = 0
    
    return x/sum(x)

def get_edge_densit_probvec(G, X, percenti=45):
    x = []
    z = gaussian_kde(X.T)(X.T)
    z = z/sum(z)

    for eij in G.edges():
        x.append(z[eij[0]]+z[eij[1]])
    
    x = np.array(x)
    cutoff = np.percentile(x, percenti)
    x[x < cutoff] = np.percentile(x, percenti/15)
    
    return x/sum(x)


#####################
num_sim  = 1
samp_col = 0.05
mult     = 1.1
sparsity = 0.98
noise    = 0.05
node_base_size = 5.5
edge_base_size = 0.6
pastl = ["#aa47d3","#78da4c","#6866d1","#ced24a","#c2479b","#79da94","#d63b45","#70d1c9","#e05729","#71a4d3",
         "#d6983d","#6b6295","#5a8e36","#d096d9","#4e8060","#ca466d","#cec98e","#ae5e43","#cb8693"]
#####################

def create_art():
    stringname = 'oxford'
    h =      9.0
    w =      h * 1.77777777778
    p =      0.05
    threshs = [11]
    # thresh = 15.75
    pix_ths = [0.90, 0.95]
    pix_th = pix_ths[0]
    thresh = threshs[0]
    sim = 0
    # for pix_th in pix_ths:
    #     for thresh in threshs:
    #         for sim in range(num_sim):
    data = plt.imread('figs/pngs/%s.png' % stringname)
    y,x = np.where(rgb2gray(data[:,:,:3]) < pix_th)
    y_norm, x_norm = map(float, data[:,:,0].shape)
    colors = data[:,:,:3]
    X0 = np.array(random.sample(list(zip(x,y)), int(len(y)*p)))*1.0
    tree0 = sp.spatial.distance.cdist(X0, X0)
    A0 = tree0 < thresh

    node_color0 = [rgb2hex(colors[int(xx), int(yy),:]) for yy, xx in X0] 
    samp = 2000
    random_percentage = np.random.choice(list(range(len(node_color0))), 
                                            int(len(node_color0)/samp),
                                            replace=False)
    colorful_nodes = []
    for i in random_percentage: 
        node_color0[i] = np.random.choice(pastl)
        colorful_nodes.append(i)

    G0 = nx.from_numpy_array(A0*1.0)
    G0.remove_edges_from(list(nx.selfloop_edges(G0)))

    nx.set_node_attributes(G0, dict(zip(list(G0.nodes()), node_color0)), 'color')
    pos0 = list(zip(*X0))
    xx0 = np.array(list(pos0[0]))    + (np.random.gamma(noise, 1, len(list(pos0[0])))\
                                        *  np.random.choice([-1,1],  len(list(pos0[0]))))
    yy0 = -1*np.array(list(pos0[1])) + (np.random.gamma(noise, 1, len(list(pos0[1])))\
                                        *  np.random.choice([-1,1],  len(list(pos0[1]))))
    pos0 = dict(zip(list(range(A0.shape[0])), list(zip(xx0, yy0))))

    ec0 = get_edge_color(G0)
    es0 = edge_base_size * mult
    nc0 = list(nx.get_node_attributes(G0, 'color').values())
    prob0 = np.array(list(dict(G0.degree()).values())) / \
        sum(np.array(list(dict(G0.degree()).values())))
    ns0 = node_base_size * ((1-prob0)**2000) * mult
    for i in range(len(ns0)):
        if node_color0[i] in pastl:
            ns0[i] = ns0[i]*1.15

    ############################### PLOT ###################################
    fig, ax = plt.subplots(figsize=(w*mult,h*mult))
    
    nx.draw_networkx_nodes(G0, pos=pos0, node_size=ns0, node_color=nc0, ax=ax, alpha=0.90)
    nx.draw_networkx_edges(G0, pos=pos0, width=es0,     edge_color=ec0, ax=ax, alpha=0.80)
    
    ax.set_axis_off()
    xlims = list(list(zip(*list(pos0.values())))[0])
    ylims = list(list(zip(*list(pos0.values())))[1])
    xpad = mult*(abs(max(xlims))+abs(min(xlims)))/150
    ypad = mult*(abs(max(ylims))+abs(min(ylims)))/150

    plt.suptitle(r'$\bf{Network\;Science,\;Network\;Visualization,\;and\;Python}$'+'\nby Brennan Klein\n2023-06-28',
                 y=0.85, ha='center', x=0.575, fontsize=24, color='.2')
    plt.xlim(min(xlims)-xpad-5.0, max(xlims)+xpad+5.0)
    plt.ylim(min(ylims)-ypad-8.0, max(ylims)+ypad+8.0)

    # plt.savefig('figs/pngs/oxford_network.png',dpi=425,bbox_inches='tight')
    plt.show()
