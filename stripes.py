#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 16:25:35 2020

@author: balthasar
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools
from pathlib import Path
from matplotlib import colors
from scipy.stats import ks_2samp
#from statsmodels.stats import multitest
from multiprocessing import Pool
import argparse
import straw
import scipy.sparse


###############################################################################
#%% ############################# parameters ##################################
###############################################################################

############################# argparse parameters #############################

#TODO: check out pathlib.PATH for basedir and other paths and use argparse type 

parser = argparse.ArgumentParser(description='Processing chromatine interactions')
parser.add_argument('bedfile',
                    help='The bedfile containing the positions to search stripes')
parser.add_argument('genebed',
                    help='The bedfile containing the gene positions')
parser.add_argument('samples', nargs="+",
                    help='The name of the samples')
parser.add_argument('--output', default="output",type=Path,
                    help='''The directory in which the output should be written''')
parser.add_argument('--stripe', default=1, type=int,
                    help='How width the stripe should be')
parser.add_argument('--resolution', default=10000, type=int,
                    help='The resolution of the input files')
parser.add_argument('--search_range', default=1000000, type=int,
                    help='''In which range interactions between genes and
                    specified sites should be investigated. If you want to
                    investigate over the complete range of all chromosomes,
                    set the number higher than the length of the longest
                    chromosome''')
#parser.add_argument('--norm', default=False, action="store_true",
#                    help='''Normalization of the hic files. If norm is present 
#                    the second hic file is scaled to the maximum value of the
#                    first file per chromosome''')
parser.add_argument('--hicnorm', default="NONE", type=str,
                    help='''which hic normalization method should be used upon
                    loading data from the hic file''')
parser.add_argument('--allbins', default=False, action="store_true",
                    help='search in all bins instead of provided dhsbins')
parser.add_argument('--combine_sample_counts', default=False, action="store_true",
                    help='''If this is true the counts in each group are added.
                    When having sparse data this might alleviate some of the 
                    gaps''')
parser.add_argument('--cores', default=8, type=int,
                    help='''How many CPU cores to use. When using this option 
                    with a value of more than one, the calculations of the
                    p-values get divided between different processes. So there
                    will be no additional speedup, when using a number higher
                    than the number of chromosomes of the samples organism''')


parser.add_argument('--minlength', default=40000, type=int,
                    help='''Gene length cutoff''')
parser.add_argument('--minsparse', default=0.2, type=float,
                    help='''Min sparsity cutoff''')
args = parser.parse_args()

############################ parameter  conversion ############################

samples = args.samples
dhsfile = args.bedfile
genebed = args.genebed
res     = args.resolution
output_dir = args.output
STRIPE_WIDTH = args.stripe
NORM = False #args.norm
COMBINE = args.combine_sample_counts
CORES  = args.cores
HICNORM = args.hicnorm


ALLBINS = args.allbins


MIN_LENGTH = args.minlength
SPARSITY_CUTOFF = args.minsparse

############################## fixed  parameters ##############################

DISTRIBUTION = False
plotoffset = 5
PVAL = True

###############################################################################
#%% ######################## Calculated Constants #############################
###############################################################################

snames = {}
for sample in samples:
    snames.setdefault(Path(sample).parent.name,[]).append(sample)

groupn = len(snames)
if groupn not in [1,2]:
    raise ValueError("""The sample directories are used to determine groups and
                     This script does not support more than two groups""")

groups = [k for k in snames]

gnames = {}
for k,vl in snames.items():
    for v in vl:
        gnames[v] = k


MARGIN = 2*STRIPE_WIDTH+1

dhsbins = {}
if not ALLBINS:
    with open(dhsfile) as f:
        for line in f:
            l = line.strip().split("\t")
            dhsbins.setdefault(l[0],[]).append(int(l[1])//res)

genebins = {}
with open(genebed) as f:
    for line in f:
        l = line.strip().split("\t")
        genebins.setdefault(l[0],[]).append((int(l[1])//res,int(l[2])//res,l[3] ))

enhancerpos = {}
with open(dhsfile) as f:
    for line in f:
        l = line.strip().split("\t")
        enhancerpos.setdefault(l[0],[]).append((int(l[1])//res,l[1],l[2]))

chromosomes = [{c.name for c in straw.getChromosomes(sample)} for sample in samples]
chrinfo = chromosomes[0].intersection(*chromosomes)
chrinfo.remove("All") # ALL seems to always exist
if chrinfo := list(chrinfo):
    print("Chromosomes present in all file:", chrinfo)
else:
    raise ValueError("No chromosome overlap")

gene_to_bin = {symbol:(start,end) for genes in genebins.values() for start, end, symbol in genes}

dhs_to_pos  = {}
for chrom in enhancerpos:
    dhs_to_pos[chrom] = {}
    for dbin,pos1,pos2 in enhancerpos[chrom]:
        dhs_to_pos[chrom].setdefault(dbin, []).append((pos1,pos2))

with open(genebed) as f:
    genesymbol_to_pos = {splitline[3]:(*splitline[:3],*splitline[4:]) for line in f if (splitline := line.strip().split("\t"))}

srange = 1000000 // res

dhs_gene_pair = {}

for chrom in chrinfo:
    dhs_gene_pair[chrom] = {}
    if chrom in genebins:
        dhsset = set()
        for gene in genebins[chrom]:
            _, gstart, gend = genesymbol_to_pos[gene[2]]
            if MIN_LENGTH < int(gend) - int(gstart):
                if ALLBINS:
                    for dhs in range(gene[0] - srange,gene[1] + srange):
                        if dhs not in dhsset:
                            dhsset.add(dhs)
                        if gene[0] - srange < dhs < gene[0]-1:
                            dhs_gene_pair[chrom].setdefault(dhs, []).append(gene[2])
                        elif gene[1] + 1 < dhs < gene[1] + srange:
                            dhs_gene_pair[chrom].setdefault(dhs, []).append(gene[2])
                else:
                    for dhs in dhsbins[chrom]:
                        if gene[0] - srange < dhs < gene[0]-1:
                            dhs_gene_pair[chrom].setdefault(dhs, []).append(gene[2])
                        elif gene[1] + 1 < dhs < gene[1] + srange:
                            dhs_gene_pair[chrom].setdefault(dhs, []).append(gene[2])
                
        if ALLBINS:
            dhsbins[chrom] = list(dhsset)

###############################################################################
#%% ############################# first loop ##################################
###############################################################################

# Iterate over each dhs gene pair and collect the interactions. Afterwards, sum
# over the interactions in one stripe and calculate the fold change between this
# and the two neighbouring stripes. Additionally a KS test is performed between
# the middle stripe and the upper and lower one.
#
# To ensure the best result and reduce the number of false positives, from each
# of the two comparisons, only the worst value for each metric is used as
# quality measure.

NEIGHBOURS = 1
TOTAL = 1+2*NEIGHBOURS

def get_sparse(sample, chrom, res):

    # MARGIN needs to be added to second coordinates so that the neighbours have offset
    CM = straw.strawC("observed", HICNORM, sample, chrom, chrom, "BP", res)
    
    x = np.zeros(len(CM),dtype=int)
    y = np.zeros(len(CM),dtype=int)
    c = np.zeros(len(CM))
    
    for i,r in enumerate(CM):
        x[i] = r.binX
        y[i] = r.binY
        c[i] = r.counts
    
    V = c
    
    I = x//res
    J = y//res
    
    R = scipy.sparse.coo_matrix((V,(I,J)),shape=(max(I)+1,max(J)+1)).tocsr()
    
    return R

def stripes_from_sparse(sparse, res, dhs, gstart, gend, STRIPE_WIDTH):
    if dhs < gstart:
        l_stripe = sparse[dhs-STRIPE_WIDTH-MARGIN:dhs-STRIPE_WIDTH,gstart-MARGIN:gend-MARGIN+1]
        c_stripe = sparse[dhs-STRIPE_WIDTH:dhs+STRIPE_WIDTH+1,gstart:gend+1]
        r_stripe = sparse[dhs+STRIPE_WIDTH+1:dhs+STRIPE_WIDTH+MARGIN+1,gstart+MARGIN:gend+MARGIN+1]
    else:
        l_stripe = sparse[gstart-MARGIN:gend-MARGIN+1,dhs-STRIPE_WIDTH-MARGIN:dhs-STRIPE_WIDTH]
        c_stripe = sparse[gstart:gend+1,dhs-STRIPE_WIDTH:dhs+STRIPE_WIDTH+1]
        r_stripe = sparse[gstart+MARGIN:gend+MARGIN+1,dhs+STRIPE_WIDTH+1:dhs+STRIPE_WIDTH+MARGIN+1]
        
    return [np.asarray(l_stripe.todense()),np.asarray(c_stripe.todense()),np.asarray(r_stripe.todense())]

def process_dhs_genes1(chrom):

    FC_dicts = {g:{} for g in snames}
    Pv_dicts = {g:{} for g in snames}
    AV_dicts = {g:{} for g in snames}
    
    FC_dict = {}
    KS_dict = {}
    
    print(chrom)
    
    sparsel =  {sample:get_sparse(sample, chrom, res) for sample in samples}
    #TODO: normalise
    if NORM:
        
        for matrix in sparsel.values():
            sparse2 = np.max(sparsel[0])/np.max(sparsel[1]) * sparsel[1]
            sparsel = [sparsel[0],sparse2]
    
    
    for dhs in dhsbins[chrom]:

        if dhs in dhs_gene_pair[chrom]:

            for genesymbol in dhs_gene_pair[chrom][dhs]:
                
                for width in range(3):
                    pass
                
                skip = False
                
                gstart,gend = gene_to_bin[genesymbol]

                dkey = f"{dhs}-{genesymbol}"
                
                if min(dhs, gstart) < 2:
                    continue
                    
                stripel = {sample:stripes_from_sparse(sparse, res, dhs, gstart, gend, STRIPE_WIDTH) for sample, sparse in sparsel.items()}
                
                if groupn > 1:
                    
                    
                    new_stripel = {}
                    for sample, l in stripel.items():
                        group = gnames[sample]
                        try:
                            if group in new_stripel:
                                new_stripel[group][0] += l[0]
                                new_stripel[group][1] += l[1]
                                new_stripel[group][2] += l[2]
                            else:
                                new_stripel[group] = [None] * 3
                                new_stripel[group][0] = l[0]
                                new_stripel[group][1] = l[1]
                                new_stripel[group][2] = l[2]
                        except ValueError:
                            # some samples fail, because they are missing values
                            # skip those
                            skip = True
                            continue

                    if skip:
                        continue
                            
                    stripel = new_stripel
                
                if not all([l[0].shape == l[1].shape == l[2].shape for l in stripel.values()]):
                    # if any group has mismatching stripes, abort
                    continue

                #print(stripel)
                
                if all([nl.size > 0 for gl in stripel.values() for nl in gl]):
                    if np.max([np.sum(np.array(nl) == 0) for gl in stripel.values() for nl in gl]) < len(stripel[groups[0]][1]) * (1-SPARSITY_CUTOFF) :
                        for group in snames:
                            
                            l_stripe, c_stripe, r_stripe = stripel[group]

                            FC_dicts[group][dkey] = np.log(np.sum(c_stripe)/max([np.sum(l_stripe),np.sum(r_stripe)]))
                            Pv_dicts[group][dkey] = max([ks_2samp(c_stripe.flatten(),l_stripe.flatten(), alternative="less")[1],
                                                     ks_2samp(c_stripe.flatten(),r_stripe.flatten(), alternative="less")[1]])
                            AV_dicts[group][dkey] = np.average(c_stripe)
                        
                        if groupn == 2:
                            FC_dict[dkey] = np.log2(np.sum(stripel[groups[0]][1]) / np.sum(stripel[groups[1]][1]))
                            KS_dict[dkey] = ks_2samp(stripel[groups[0]][1].flatten(),stripel[groups[1]][1].flatten(), alternative="less")[1]

    return FC_dicts,Pv_dicts,AV_dicts, FC_dict, KS_dict


# remove chromosomes where no dhs are present
keylist = {key for key in chrinfo}

for key in dhsbins:
    if key in keylist:
        keylist.remove(key)

for key in keylist:
    chrinfo.remove(key)

if CORES <=1:
    results = [process_dhs_genes1(chrom) for chrom in chrinfo]

else:
    with Pool(processes=CORES) as pool:
        results = pool.map(process_dhs_genes1, chrinfo)
#TODO: use imap_unordered istead of map for unordered output but maybe faster


FC_WT_dict = {}
KSpvaldict = {}
AV_WT_dict = {}

FC_dict = {}
KS_dict = {}

if groupn == 2:
    FC_DRBdict = {}
    KSpDRBdict = {}
    AV_DRBdict = {}

for result in results:
    FC_WT_dict = FC_WT_dict | result[0][groups[0]]
    KSpvaldict = KSpvaldict | result[1][groups[0]]
    AV_WT_dict = AV_WT_dict | result[2][groups[0]]
    FC_dict    = FC_dict | result[3]
    KS_dict    = KS_dict | result[4]
    
    if groupn == 2:
        FC_DRBdict = FC_DRBdict | result[0][groups[1]]
        KSpDRBdict = KSpDRBdict | result[1][groups[1]]
        AV_DRBdict = AV_DRBdict | result[2][groups[1]]

#%% ############################# select plot #################################

# Selection of the most significant DHS gene interactions. Different methods
# include the pvalue of the KS test and the raw fold change.
# The selected sites are put into a list
#
# Additionally the number of significant sites for control and treatment are
# counted and compared


sortdict = KSpvaldict
items = []
keys  = []

for k, i in sortdict.items():
    if not np.isnan(i) and np.abs(i) != np.inf:
        if KS_dict[k] < 0.05:
            keys.append(k)
            items.append(i)


arg = np.argsort(items)
items = np.array(items)[arg]
key = np.array(keys)[arg]

top20 = list(key[:20])

# if groupn > 1:
    
#     sortdict2 = KSpDRBdict
    
#     items2 = []
#     keys2  = []
    
#     for k, i in sortdict2.items():
#         if not np.isnan(i) and np.abs(i) != np.inf:
#             keys2.append(k)
#             items2.append(i)
    
    
#     arg2 = np.argsort(items2)
#     items2 = np.array(items2)[arg2]
#     key2 = np.array(keys2)[arg2]
    
    
#     #_, pvals,_,_ = multitest.multipletests(items, method="bonferroni")
    
#     #KSpvaldict_corrected = dict(zip(key,pvals))
#     #KSpDRBdict_corrected = dict(zip(KSpDRBdict.keys(),multitest.multipletests(list(KSpDRBdict.values()), method="bonferroni")[1]))
    
#     top20 = list(key[:10]) + list(key2[:10])

# else:
#     top20 = list(key[:20])

###############################################################################
#%% ############################# second loop #################################
###############################################################################

# In the second loop the interaction values for each site, selected for plotting
# are collected into a dictionary. The biggest time cost in this step is the
# loading of the sparse matrix. To reduce unnecessary loops, only the chromosomes
# with selected interactions are loaded

dhscounts = {}

topchrom = {genesymbol_to_pos[dkey.split("-",maxsplit=1)[-1]][0] for dkey in top20}

def get_sparse_region(sample, chrom, start, end, res):

    # MARGIN needs to be added to second coordinates so that the neighbours have offset
    CM = straw.strawC("observed", HICNORM, sample, chrom + f":{start}:{end}", chrom + f":{start}:{end}", "BP", res)
    
    x = np.zeros(len(CM),dtype=int)
    y = np.zeros(len(CM),dtype=int)
    c = np.zeros(len(CM))
    
    for i,r in enumerate(CM):
        x[i] = r.binX
        y[i] = r.binY
        c[i] = r.counts
    
    V = c
    
    I = x//res
    J = y//res
    
    
    I -= min(I)
    J -= min(J)
    
    shape = (end - start) // res +1
    
    R = np.asarray(scipy.sparse.coo_matrix((V,(I,J)),shape=(shape,shape)).todense())
    
    return R


for dkey in top20:

    dhs = int(dkey.split("-")[0])

    genesymbol = dkey.split("-",maxsplit=1)[1]
    chrom = genesymbol_to_pos[dkey.split("-",maxsplit=1)[-1]][0]
    print(chrom)
    
    gstart,gend = gene_to_bin[genesymbol]
    start = min(dhs,gstart)
    end   = max(dhs,gend)

    if dhs:

        extension = 2 * plotoffset + 2 + int(0.5 *(end-start))
        counts = [get_sparse_region(sample, chrom, (start - extension)*res, (end + extension)*res, res) for sample in samples]
        
        
        #TODO: normalise
        if NORM:
            counts = [counts[0],np.max(counts[0])/np.max(counts[1]) * counts[1]]
        
        newstart = start - extension
        newdhs = dhs - newstart
        newgstart = gstart - newstart
        newgend = gend - newstart

        dhscounts[dkey] = ((newdhs,newgstart,newgend),counts) #(np.array(binlist),np.array(countlist))

###############################################################################
#%% ################################ files ####################################
###############################################################################

################################## write tsv ##################################

# In this step the data for each site, significant or not, is written into a tsv
# The data includes chromosome and position, but also the values calculated in
# the first loop, like fold change and pvalue of the KS test.

#snames = [Path(samples[spg*g]).parent.name for g in range(groupn)]

(output_dir / "genes").mkdir(parents=True, exist_ok=True)

with open( output_dir / "genes/data.tsv", "w") as f:

    header = ["chromosome",
              "genesymbol",
              "genepos1",
              "genepos2",
              "dhspos1",
              "dhspos2",
              "dhsbin",
              f"logFC {groups[1]}",
              f"pval {groups[1]}",
              f"avg intensity {groups[1]}"]
    
    for g in groups:
        header += [f"logFC {g}",
                   f"pval {g}",
                   f"avg intensity {g}"]
    
    f.write("\t".join(header)+"\n")
    
    
    
    
    for dkey in FC_WT_dict:
        dhs,genesymbol = dkey.split("-", maxsplit=1)
        chrom, genepos1,genepos2 = genesymbol_to_pos[genesymbol]
        strand = 1

        if int(dhs) in dhs_to_pos[chrom]:
            
#TODO: add dhs_gene_interactions_intensity avg 

            for p1,p2 in dhs_to_pos[chrom][int(dhs)]:
                
                line = [chrom,
                        genesymbol,
                        str(genepos1),
                        str(genepos2),
                        str(p1),
                        str(p2),
                        dhs,
                        str(round(FC_WT_dict[dkey],3)),
                        str(KSpvaldict[dkey]),
                        str(AV_WT_dict[dkey])]
                
                if groupn == 2:
                    line += [str(round(FC_DRBdict[dkey],3)),
                             str(KSpDRBdict[dkey]),
                             str(AV_DRBdict[dkey])]
                
                
                
                f.write("\t".join(line)+"\n")
            
        else:

            line = [chrom,
                    genesymbol,
                    str(genepos1),
                    str(genepos2),
                    "",
                    "",
                    dhs,
                    str(round(FC_WT_dict[dkey],3)),
                    str(KSpvaldict[dkey]),
                    str(KSpvaldict[dkey])]            

            if groupn == 2:
                line += [str(round(FC_DRBdict[dkey],3)),
                         str(KSpDRBdict[dkey]),
                         str(AV_DRBdict[dkey])]

            f.write("\t".join(line)+"\n")

###############################################################################
#%% ################################ plots ####################################
###############################################################################

# This plots the interaction matrix for the region containing the dhs and the
# gene in a triangle style plot. On the right side of the plot are annotations
# for the intensity scale. Below the plot is an annotation for the genomic
# features
#

#plt.ioff() # don't show, just save
################################ legend colors ################################

cmap   = colors.ListedColormap(['white', 'blue', 'orange'])
bounds = [-0.5, 0.5,1.5, 2.5]
norm   = colors.BoundaryNorm(bounds, cmap.N)

heatcmap = plt.cm.YlOrRd.copy()
heatcmap.set_under("w")

################################## main loop ##################################


for dkey, ((dhs,newgstart,newgend),counts) in dhscounts.items():

################################### variables #################################

    print(dkey)

    dhsbin = int(dkey.split("-")[0])

    genesymbol = dkey.split("-",maxsplit=1)[1]

    gene_binstart,gene_binend = gene_to_bin[genesymbol]

    chromosome, genepos1,genepos2 = genesymbol_to_pos[genesymbol]
    chrom = chromosome

    strand = 1

    minbin = min(gene_binstart, dhsbin) - min(dhs,newgstart)
    maxbin = max (gene_binend, dhsbin) - (len(counts) - max(dhs,newgend) )

    start, mid, end = sorted((dhs,newgstart,newgend))

    min_distance  = min(abs(newgstart - dhs),abs(newgend - dhs))
    
    stripe_length = end - start

    fullmtx = counts[0]
    
    if fullmtx.shape[0] != fullmtx.shape[1]:
        print("SHAPE MISMATCH!", fullmtx.shape, ", skipping")
        continue
    
    if groupn > 1:
        fullmtx2 = counts[1]
        if fullmtx.shape != fullmtx2.shape:
            continue

    figstart = start - plotoffset
    figend   = end   + plotoffset


    selecty = np.array([min_distance+STRIPE_WIDTH,
                        min_distance-STRIPE_WIDTH-1,
                        stripe_length-STRIPE_WIDTH,
                        stripe_length+STRIPE_WIDTH+1,
                        min_distance+STRIPE_WIDTH])

    if dhs > newgend:
        selectx = np.array([end + STRIPE_WIDTH/2 - min_distance/2 + 1,
                    end - STRIPE_WIDTH/2 - min_distance/2 + 0.5,
                    start - STRIPE_WIDTH/2 + stripe_length/2,
                    start + STRIPE_WIDTH/2 + stripe_length/2  +0.5,
                    end + STRIPE_WIDTH/2 - min_distance/2 + 1])

        stripe = fullmtx[newgstart:newgend+1,dhs-STRIPE_WIDTH:dhs+STRIPE_WIDTH+1]

    else:
        selectx = np.array([start - STRIPE_WIDTH/2 + min_distance/2,
                    start + STRIPE_WIDTH/2 + min_distance/2  +0.5,
                    start + STRIPE_WIDTH/2 + stripe_length/2 + 1,
                    start - STRIPE_WIDTH/2 + stripe_length/2 +0.5,
                    start - STRIPE_WIDTH/2 + min_distance/2])

        stripe = fullmtx[dhs-STRIPE_WIDTH:dhs+STRIPE_WIDTH+1,newgstart:newgend + 1]

    lowerx = selectx - MARGIN
    upperx = selectx + MARGIN

    #vmax = 2 * np.ceil(np.average(fullmtx[fullmtx!=0]))
    vmax = np.min([np.ceil(np.max(stripe)), 2 * np.ceil(np.average(stripe))])

############################### rotation matrix ###############################

    n = fullmtx.shape[0]
    t = np.array([[1,0.5],[-1,0.5]])
    A = np.dot(np.array([(i[1],i[0]) for i in itertools.product(range(n,-1,-1),range(0,n+1,1))]),t)

################################ figure  start ################################

    fig = plt.figure(figsize=(10,12))

    gs = fig.add_gridspec(nrows=8, ncols=2,
                          width_ratios  = [20,3],
                          height_ratios = [10,10,1,2,10,10, 1, 1],
                          hspace= 0.01,
                          wspace= 0.1,
                          left=0.01,right=0.99, top=0.95)

    fig.suptitle(f"{chromosome} {dhsbin} {genesymbol}")

#################################### axis1 ####################################

    if groupn == 1:
        heatax = fig.add_subplot(gs[:6,0])
    else:
        heatax = fig.add_subplot(gs[:2,0])

    img = heatax.pcolormesh(A[:,1].reshape(n+1,n+1),A[:,0].reshape(n+1,n+1),np.flipud(fullmtx), vmin=0.01, vmax=vmax, cmap=heatcmap)
    #img = heatax.pcolormesh(fullmtx, cmap="hot",vmin=0,vmax=5)#, vmin = -extend, vmax = extend, rasterized=True)
    heatax.set_title(f"{groups[0]} pval = {KSpvaldict[dkey]:{'.2E' if KSpvaldict[dkey] < 0.01 else '.03'}}")


    cax = fig.add_subplot(gs[1:5,-1])
    cbar = plt.colorbar(img, cax=cax)
    cax.set_title("Interaction frequency")
    cax.set_aspect(10)

    #heatax.set_title(f"Control logFC: {FC_WT_dict[dkey]:.03} pval = {KSpvaldict_corrected[dkey]:{'.2E' if KSpvaldict_corrected[dkey] < 0.01 else '.03'}}")

    heatax.set_yticks([])

    xticks = [start+0.5, end+0.5]
    xticklabels = [start + minbin, end + minbin]

    xld = fullmtx.shape[0] // 20
    #print(length, xld, min_distance)

    if stripe_length > 2*xld and stripe_length - min_distance >= xld:
        xticks.append(mid+0.5)
        xticklabels.append(mid + minbin)

    heatax.set_xticks(xticks) #[0.5 + i for i in range(length)]
    heatax.set_xticklabels(xticklabels)
    heatax.set_xlim((figstart,figend))
    heatax.set_ylim((0,(figend - figstart)))

    heatax.set_aspect(0.5)

    heatax.plot(selectx,selecty, color="g")
    heatax.plot(lowerx,selecty, color="r")
    heatax.plot(upperx,selecty, color="r")
    #heatax.grid()


#################################### axis2 ####################################


    if groupn > 1:

        heatax2 = fig.add_subplot(gs[4:6,0], sharex=heatax)

        img = heatax2.pcolormesh(A[:,1].reshape(n+1,n+1),A[:,0].reshape(n+1,n+1),np.flipud(fullmtx2), vmin = 0.01, vmax=vmax, cmap=heatcmap)

        heatax2.set_title(f"{groups[1]} pval = {KSpDRBdict[dkey]:{'.2E' if KSpDRBdict[dkey] < 0.01 else '.03'}}")

        heatax2.set_yticks([])

        heatax2.set_aspect(0.5)
        heatax2.plot(selectx,selecty, color="g")
        heatax2.plot(lowerx,selecty, color="r")
        heatax2.plot(upperx,selecty, color="r")

        #heatax2.set_xticklabels(xticklabels)
        heatax2.set_ylim((0,(figend - figstart)))

#################################### axis3 ####################################

    #dhsposstring = "DHS position" + ("s:\n" if len(dhs_to_pos[chrom][dhsbin]) > 1 else ":\n") \
    #    + "\n".join([str(p1)+"-"+str(p2) for p1,p2 in dhs_to_pos[chrom][dhsbin]])

    dhsposstring = ""
    dhsposstring += f"\nGene position:\n{genepos1}-{genepos2}\n{'+'if strand == 1 else '-'} strand"

    textax = fig.add_subplot(gs[0,-1])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textax.text(0, 0.95, dhsposstring, transform=textax.transAxes, fontsize=8, verticalalignment='top', bbox=props)
    textax.axis("off")

#################################### axis3 ####################################

    annotax = fig.add_subplot(gs[-2,0], sharex=heatax)

    annotbar = np.array([ 1 if (i == dhsbin) else (2 if gene_binstart <= i <= gene_binend else 0) for i in range(minbin, maxbin)])
    annotbar = annotbar.reshape((1,-2))
    annotax.margins(x=0)
    annotax.set_yticks(())
    #annotax.set_xticks(())
    annotax.axis("off")

    #annotax.set_title("target DHS")

    annotax.set_aspect(0.5)

    img = annotax.pcolormesh(annotbar, cmap=cmap, norm=norm)

    cax = fig.add_subplot(gs[-2:,-1])

    cbar = plt.colorbar(img, boundaries=bounds, ticks=([0,1,2]), cax=cax, orientation="horizontal")
    cbar.ax.set_xticklabels(["N/A","DHS","gene"], fontsize=8)
    cbar.ax.xaxis.set_tick_params(labeltop='on',labelbottom=False)
    cax.xaxis.set_tick_params(length=0)
    cax.set_title("Annotation legend")

#################################### axis3 ####################################

    if groupn > 1:

        annotax = fig.add_subplot(gs[2,0], sharex=heatax)
    
        annotax.margins(x=0)
        annotax.set_yticks(())
        annotax.axis("off")
    
        annotax.set_aspect(0.5)
        img = annotax.pcolormesh(annotbar, cmap=cmap, norm=norm)

#################################### dhsax ####################################

    d2ax = fig.add_subplot(gs[-1,0], sharex=heatax)

    d2arr = np.array([ 1 if (i in dhsbins[chrom]) else 0 for i in range(minbin, maxbin)])
    d2arr = d2arr.reshape((1,-2))
    d2ax.margins(x=0)
    d2ax.set_yticks(())
    #d2ax.set_xticks(())
    d2ax.axis("off")

    #d2ax.set_title("all DHS")

    img = d2ax.pcolormesh(d2arr, cmap=cmap, norm=norm)

#################################### axis4 ####################################

    #dhsax = fig.add_subplot(gs[-1,0], sharex=heatax)

    #dhsax.margins(x=0,y=1)
    #dhsax.set_yticks(())
    #dhsax.set_xticks(())

    #dhsax.set_title("pvalue of all DHS")

    #dhsax.set_xlim((figstart,figend))
    #dhsax.set_ylim((0,np.max(dhsarr)+1))

    #dhsax.axis("off")

    #dhsax.plot(dhsarr)
    #dhsax.fill_between([i for i in range(len(dhsarr))],dhsarr)

#################################### save ####################################
    
    ( output_dir / "genes/").mkdir(parents=True, exist_ok=True)
    fig.savefig( output_dir / "genes/{dkey}_triangle.pdf", bbox_inches='tight')
    #fig.savefig(f"aggregate_figures/genes/{dkey}_triangle.png", bbox_inches='tight')
    
    
    if dhs > newgend:

        l_stripe = fullmtx[newgstart-MARGIN:newgend-MARGIN+1,dhs-STRIPE_WIDTH-MARGIN:dhs-STRIPE_WIDTH]
        c_stripe = fullmtx[newgstart:newgend+1,dhs-STRIPE_WIDTH:dhs+STRIPE_WIDTH+1]
        r_stripe = fullmtx[newgstart+MARGIN:newgend+MARGIN+1,dhs+STRIPE_WIDTH+1:dhs+STRIPE_WIDTH+MARGIN+1]

    else:
    
        
        l_stripe = fullmtx[dhs-STRIPE_WIDTH-MARGIN:dhs-STRIPE_WIDTH,newgstart-MARGIN:newgend-MARGIN+1]
        c_stripe = fullmtx[dhs-STRIPE_WIDTH:dhs+STRIPE_WIDTH+1,newgstart:newgend+1]
        r_stripe = fullmtx[dhs+STRIPE_WIDTH+1:dhs+STRIPE_WIDTH+MARGIN+1,newgstart+MARGIN:newgend+MARGIN+1]
    
    def ecdf(a):
        x, counts = np.unique(a, return_counts=True)
        cusum = np.cumsum(counts)
        return x, cusum / cusum[-1]
    
    
    # fig_ecdf = plt.figure(figsize=(10,12))
    # ax = fig_ecdf.add_subplot(311)
    
    # x,y = ecdf(l_stripe)
    # ax.plot(x,y, label="left")
    # x,y = ecdf(c_stripe)
    # ax.plot(x,y, label="center")
    # x,y = ecdf(r_stripe)
    # ax.plot(x,y, label="right")
    # ax.set_title("eCDF")
    # plt.legend()
    
    
    # ax = fig_ecdf.add_subplot(212)
    
    # ax.plot(np.sort(l_stripe.flatten()), label="left")
    # ax.plot(np.sort(c_stripe.flatten()), label="center")
    # ax.plot(np.sort(r_stripe.flatten()), label="right")
    # ax.set_title("sorted values")
    # plt.legend()
    
    
    # kl_l = kl_div(c_stripe,l_stripe)
    # kl_r = kl_div(c_stripe,r_stripe)
    
    print(KSpvaldict[dkey])
    # print(np.sum(kl_l[kl_l != np.inf])/ kl_l.size)
    # print(np.sum(kl_r[kl_r != np.inf])/ kl_r.size)
    # print(1 / np.min([np.sum(kl_l[kl_l != np.inf])/ kl_l.size,np.sum(kl_r[kl_r != np.inf])/ kl_r.size]))

    # fig_ecdf.savefig(f"{output_dir}/genes/{dkey}_cdf.pdf", bbox_inches='tight')

    # 1/0
    #plt.close("all")
    
    # genesymbol = "SHC4"
    # sample = samples[0]
    # chrom = genesymbol_to_pos[genesymbol][0]
    # sparse = get_sparse(sample, chrom, res)
    
    # gstart,gend = gene_to_bin[genesymbol]
    # st = stripes_from_sparse(sparse, res, 4879, gstart, gend, STRIPE_WIDTH)
    
################################# distribution ################################

    # if DISTRIBUTION:
    #
    #     stripdict = 0
    #     distfig = plt.figure()
    #     ax1 = distfig.add_subplot(121)

    #     ax1.plot(np.cumsum(np.sort(stripdict[dkey])))

    #     ax2 = distfig.add_subplot(122)
    #     ax2.hist(stripdict[dkey],bins=list(range(21)))

    #     distfig.savefig(f"aggregate_figures/genes/{dkey}_distribution.pdf", bbox_inches='tight')
print( output_dir / "genes/" )
