
import sys
import math
import itertools
import operator
import logging

import numpy
# cKDTree does not supoort
# query ball
from scipy.spatial import KDTree
from tqdm.autonotebook import tqdm

_logger = logging.getLogger('gmatch')


import math
import collections
import itertools
import logging

import numpy as np

_logger = logging.getLogger('gmatch')

Triangle = collections.namedtuple('Triangle', ['v0', 'v1', 'v2', 'i0', 'i1', 'i2', 'logp', 'hel', 'R', 'tR', 'C', 'tC'])

MatchedTriangles = collections.namedtuple('MatchedTriangles', ['t0', 't1', 'hel', 'logm'])

def norma(x):
    n = np.sqrt(np.dot(x, x.conj()))
    return n

def votes(matches, c1, c2):
    # shape of the catalogues, not of the matches
    vot = np.zeros((c1, c2), dtype='int')
    # to store matched points  
    lm1 = []
    lm2 = []

    for m in matches:
        t0 = m.t0
        t1 = m.t1

        vot[t0.i0, t1.i0] += 1
        vot[t0.i1, t1.i1] += 1
        vot[t0.i2, t1.i2] += 1

    vmx = vot.max()
    _logger.debug('maximum voting count %i', vmx)
    if vmx <= 0:
        _logger.info('voting is 0, no match between catalogues')
        return np.array([[]])
    
    sortv = np.argsort(vot, axis=None)
    id0, id1 = np.unravel_index(sortv[::-1], (c1, c2))
    for i, j in zip(id0, id1):
        val = vot[i, j]
        if val <= 0:
            # votes are 0
            _logger.info('votes have reached 0 level, ending')
            break
            
        if 2 * val < vmx:
            _logger.info('votes are a half of the maximum, ending')
            # votes are a half of the maximum
            break
        if (i in lm1) or (j in lm2):
            # the point is already matched
            _logger.info('point %i %i already matched, ending', i, j)
            break

        _logger.debug('obj %i in cat1 is matched with obj %i in cat2', i, j)
        lm1.append(i)
        lm2.append(j)

    result = np.array([lm1, lm2]).T
    return result

def _scale_factor(mf, mt):
    if mf > mt:
        return 1
    elif 0.1 * mt > mf:
        return 3
    else:
        return 2

def clean_matches(matches):

    while True:
        nmatches = len(matches)
        npl = nm = 0
        logm = []
        for match in matches:
            if match.hel > 0:
                npl += 1
            elif match.hel < 0:
                nm += 1
            else:
                _logger.error('hel must not be 0')
                break
            logm.append(match.logm)

        _logger.debug('n+ is %i', npl)
        _logger.debug('n- is %i', nm)

        mt = abs(npl - nm)
        mf = npl + nm - mt

        scale = _scale_factor(mf, mt)
        _logger.debug('scale factor is %f', scale)

        lgmrr = np.array(logm)

        med = lgmrr.mean()
        std = lgmrr.std()

        _logger.debug('log M, average=%g std=%g', med, std)

        if std == 0:
            _logger.debug('std is 0, end matching')
            break

        removed_matches = 0
        newmatches = []
        _logger.info('removing false matches due to scale')
        for match in matches:
            z = (match.logm - med ) / (scale * std)
            if -1 <= z <= 1:
                newmatches.append(match)
            else:
                removed_matches += 1

        _logger.info('matches were %i', nmatches)
        _logger.info('rejected matches %i', removed_matches)
        _logger.info('matches are %i', nmatches - removed_matches)

        matches = newmatches
        if removed_matches == 0:
            _logger.info('finished filtering')
            break

    return matches

def match_triangs(t1, lt):

    def mr(t1, t2):
        return (t1.R - t2.R)**2 - t1.tR**2 - t2.tR**2

    def mc(t1, t2):
        return (t1.C - t2.C)**2 - t1.tC**2 - t2.tC**2

    def distance(t1, t2):
        return (t1.R - t2.R)**2 + (t1.C - t2.C)**2

    matched = []

    for t2 in lt:
        sen1 = mr(t1, t2)

        if sen1 > 0:
            continue

        sen2 = mc(t1, t2)

        if sen2 > 0:
            continue

        matched.append((distance(t1, t2), t2))

    _logger.debug('trg has %i close neighbors', len(matched))

    if not matched:
        return None

    dm, tm = min(matched)
    _logger.debug('closest triangle is %s', tm)
    _logger.debug('distance is %g', dm)

    return MatchedTriangles(t1, tm, t1.hel * tm.hel, t1.logp - tm.logp)

def match_triang(t1, t2):

    def mr(t1, t2):
        return (t1.R - t2.R)**2 - t1.tR**2 - t2.tR**2
    def mc(t1, t2):
        return (t1.C - t2.C)**2 - t1.tC**2 - t2.tC**2

    sen1 = mr(t1, t2)

    if sen1 > 0:
        return None

    sen2 = mc(t1, t2)

    if sen2 > 0:
        return None

    return MatchedTriangles(t1, t2, t1.hel * t2.hel, t1.logp - t2.logp)
    

def create_triang(vlist, reject_scale=10, ep=1e-3):
    for idx in tqdm(itertools.combinations(range(vlist.shape[0]), 3)):
        t = create_triang_(vlist, idx, ep)
        if t.R < reject_scale:
            yield t

def create_triang_(vlist, idx, ep=1e-3):
    v = vlist[idx, :]
    # sides
    # np.roll(v[:,0:2],shift=-1,axis=0)
    a = v[[1,2,0], 0:2] - v[:, 0:2] # 1-0, 2-1, 0-2
    # norms of the sides
    n = [norma(ar) for ar in a]
    # perimeter
    p = sum(n)
    ll = [(ni, ai, ids, (ids + 1) % 3) for ni, ai, ids in zip(n, a, range(3))]
    ls = sorted(ll)
    sides, aristas, idxs, nidxs = zip(*ls)

    ov = v[idxs, :]
    oa = np.array(aristas)

    # cross product of sides
    e = np.cross(oa, np.roll(oa, shift=-1, axis=0))

    sg = np.sign(e)
    if np.any(sg != sg[0]):
        _logger.info('reorder')
    R = sides[2] / sides[0]
    C = np.dot(oa[0], oa[2]) / (sides[2] * sides[0])
    dep1 = (1.0 / (sides[2])**2 + 1.0 / sides[0]**2 - C / (sides[2] * sides[0]))
    tR = 2 * R * R * ep * ep * dep1
    tC = 2 * (1 - C**2) * ep**2 * dep1 + 3 * C**2 * ep**4 * dep1**2

    return Triangle(v[0], v[1], v[2], idx[0], idx[1], idx[2], math.log(p), sg[0], R, tR, C, tC)

def gmatch(cat1, cat2, reject_scale=10.0, eps=1e-3):

    step = 0
    maxstep = 3
    ic1 = cat1
    ic2 = cat2
    while True:
        _logger.info('number of points cat1 %i', len(ic1))
        _logger.info('number of points cat2 %i', len(ic2))
        maxmatch = min(ic1.shape[0], ic2.shape[0])
        if maxmatch < 1:
            _logger.warning('no possible matches between catalogues')
            return None
        _logger.info('maximum number of matches %i', maxmatch)
        pm = gmatch_once(ic1, ic2, reject_scale, eps)

        nmatches = len(pm)
        _logger.info('matches: %i, maximum matches: %i', nmatches, maxmatch)
        if nmatches == maxmatch:
            _logger.info('objects in current input catalogues are all matched in %i steps', step)
            return ic1[pm[:,0]], ic2[pm[:,1]]

        elif nmatches == 0:
            _logger.info('no matches in catalogues')
            return None
        elif nmatches < maxmatch:
            if step < maxstep:
                _logger.info('number of matches < number of maximum matches')
                _logger.info('starting over with only matched points')             
                
                ic1 = cat1[pm[:,0]]
                ic2 = cat2[pm[:,1]]
                step += 1
            else:
                _logger.info('no convergence in matches after %i steps', maxstep)
                return None
                
        else:
            _logger.error('something is very wrong')
            return None


def gmatch_once(cat1s, cat2s, reject_scale=10.0, eps=1e-3):
    _logger.info('generating triangles in catalogue 1')
    tl1 = list(create_triang(cat1s, reject_scale=reject_scale, ep=eps))
    c = cat1s.shape[0]
    _logger.info('expected triangles %i', c * (c - 1) * (c - 2 ) / 6)
    _logger.info('created triangles %i', len(tl1))

    _logger.info('generating triangles in catalogue 2')
    tl2 = list(create_triang(cat2s, reject_scale=reject_scale, ep=eps))
    c = cat2s.shape[0]
    _logger.info('expected triangles %i', c * (c - 1) * (c - 2 ) / 6)
    _logger.info('created triangles %i', len(tl2))

    mrt1 = max(tl1, key=operator.attrgetter('tR'))
    mct1 = max(tl1, key=operator.attrgetter('tC'))
    _logger.debug('max R tolerance 1 %f', mrt1.tR)
    _logger.debug('max C tolerance 1 %f', mct1.tC)
    mrt2 = max(tl2, key=operator.attrgetter('tR'))
    mct2 = max(tl2, key=operator.attrgetter('tC'))
    _logger.debug('max R tolerance 2 %f', mrt2.tR)
    _logger.debug('max C tolerance 2 %f', mct2.tC)

    maxR = math.sqrt(mrt1.tR**2 + mrt2.tR**2)
    maxC = math.sqrt(mct1.tC**2 + mct2.tC**2)
    maxdis = math.sqrt(maxR**2 + maxC**2)
    _logger.info('max query tolerance in R space %f', maxR)
    _logger.info('max query tolerance in C space %f', maxC)
    _logger.info('max query tolerance in R-C space %f', maxdis)

    _logger.info('spliting R and C in catalogues')
    tspace1 = numpy.array([[tl.R, tl.C] for tl in tl1])    
    tspace2 = numpy.array([[tl.R, tl.C] for tl in tl2])

    _logger.info('finding closer triangles')
    # We use a kD tree to search in the R-C space
    _logger.debug('create kdtree...')
    kdtree = KDTree(tspace2)
    _logger.debug('done')

    _logger.debug('query in tree...')
    r = kdtree.query_ball_point(tspace1, r=maxdis)
    # r is an array of lists
    _logger.debug('done')
    
    matches1 = []
    _logger.info('checking matches')
    for i, l in numpy.ndenumerate(r):
        i0 = i[0]
        # triangle in first catalogue
        t1 = tl1[i0]
        # list of triangles in secodn catalogue
        t2s = [tl2[i1] for i1 in l]
        _logger.debug('trg %i in cat1 has %i neighbours', i0, len(t2s))
        # Return the closets trg in R-C space
        # or None
        mm = match_triangs(t1, t2s)
        if mm:
            matches1.append(mm)
    
    _logger.info('we have %i matches', len(matches1))
    if not matches1:
        _logger.info('no matches between the catalogues')
        return []
    
    # removing false matches
    _logger.info('filtering matches')
    matches = clean_matches(matches1)
    # voting to select matched coordinates
    _logger.info('voting')
    pm = votes(matches, cat1s.shape[0], cat2s.shape[0])    
    return pm