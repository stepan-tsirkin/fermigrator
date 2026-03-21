import numpy as np



def intersections(contour1, contour2):
    """Find the intersection points of trwo contours. Each counour ig geven by its 
    segments (start and end points) and does NOT have to be continuous, so thew segments order is not important. 
    
    Parameters
    ----------
    contour1, contour2 : np.array
        An array of shape (N, 2, 2) where each row represents a segment of the contour in the format [[x1, y1],[ x2, y2]].
    
    Returns
    -------
    list of tuples (i,j, x, y)
        A list of tuples where each tuple contains the indices of the segments from contour1 and contour2 that intersect, and the coordinates of the intersection point (x, y).
    """
    c1min = contour1.min(axis=1)
    c1max = contour1.max(axis=1)
    c2min = contour2.min(axis=1)
    c2max = contour2.max(axis=1)
    candidate_pairs = []
    for i in range(len(contour1)):
        select = np.where((c2max[:, 0] >= c1min[i, 0]) & (c2min[:, 0] <= c1max[i, 0]) &
                          (c2max[:, 1] >= c1min[i, 1]) & (c2min[:, 1] <= c1max[i, 1]))[0]
        for j in select:
            candidate_pairs.append((i, j))
    intersections = []
    for i, j in candidate_pairs:
        p1, p2 = contour1[i]
        p3, p4 = contour2[j]
        denom = (p4[1] - p3[1]) * (p2[0] - p1[0]) - (p4[0] - p3[0]) * (p2[1] - p1[1])
        if denom == 0:
            continue  # Parallel lines
            # note, that parallel lines can still intersect if they are collinear, but we ignore this case for now
            # it can actually lead to singularities in QPI
        ua = ((p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0])) / denom
        ub = ((p2[0] - p1[0]) * (p1[1] - p3[1]) - (p2[1] - p1[1]) * (p1[0] - p3[0])) / denom
        if 0 <= ua <= 1 and 0 <= ub <= 1:
            x = p1[0] + ua * (p2[0] - p1[0])
            y = p1[1] + ua * (p2[1] - p1[1])
            intersections.append((i, j, x, y))
    return intersections

def cut_boundary(contour, indices, axis, side):
    # by construction, the start point of each segment is within the unit square, so we only need to check the end point of each segment
    shift = np.zeros(2)
    epsilon = 1e-7
    if side == "lower":
        select = np.where(contour[:, 1, axis] < -epsilon)[0]
        boundary_value = 0
        shift[axis] = 1
    elif side == "upper":
        select = np.where(contour[:, 1, axis] > 1+epsilon)[0]
        boundary_value = 1
        shift[axis] = -1
    else:
        raise ValueError(f"Invalid side: {side}")
    new_segments = []
    new_indices = []
    for i in select:
        index_original = indices[i]
        p1, p2 = contour[i].copy()
        t = (boundary_value - p1[axis]) / (p2[axis] - p1[axis])
        assert 0 <= t < 1
        intersection_point = p1 + t * (p2 - p1)
        contour[i, 1, :] = intersection_point
        new_segments.append([intersection_point+shift, p2+shift])
        new_indices.append(index_original)
    if len(new_segments) > 0:
        indices = np.hstack((indices, new_indices))
        new_segments = np.array(new_segments)
        contour = np.concatenate((contour, new_segments), axis=0)
    return contour, indices



def shift_contour_mod1(contour, q):
    """Shift the contour points to be within the unit square [0, 1] x [0, 1] by applying modulo 1 to the coordinates.
    If some segments of the contour cross the boundary of the unit square, they will be split into multiple segments that lie within the unit square.
    
    Parameters
    ----------
    contour : np.array
        An array of shape (N, 2, 2) where each row represents a segment of the contour in the format [[x1, y1],[ x2, y2]].
    q : np.array (2,)
        A 2D vector that represents the shift to be applied to the contour points.
    
    Returns
    -------
    np.array (M, 2, 2)
        An array of shape (M, 2, 2) where each row represents a segment of the shifted contour in the format [[x1, y1],[ x2, y2]]. 
        M >= N, as some segments may be split into multiple segments if they cross the boundary of the unit square.
    np.array (M, dtype=int)
        Indices if the original segments that correspond to the shifted segments. 
        Starts with 0,1,..N-1, and if a segment is split into multiple segments, the same index will be repeated for those segments.
    """
    contour = np.copy(contour)
    contour = contour + q
    contour_start_mod1 = contour[:, 0] % 1
    shift = contour_start_mod1 - contour[:, 0]
    contour += shift[:, None, :]
    indices = np.arange(len(contour))

    for axis in 0, 1:
        for side in ["lower", "upper"]:
            contour, indices = cut_boundary(contour, indices, axis, side)
    return contour, indices

    
