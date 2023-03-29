import numpy as np


def local2world(position, prev_pos, data, idx_elem):
    """
    Compute absolute positions for EML on X,Y
    :param position:
    :param prev:
    :param data: segment data provided in USK
    :param idx_elem: 0 - EML X, 1 - EML Y
    :return:
    """
    position_world = []
    for idx in range(len(data)):
        cur_pos = data[idx][idx_elem]
        delta = cur_pos - prev_pos
        compensate = 0
        if delta < -8:
            compensate = 16
        elif delta > 8:
            compensate = -16
        position += (delta + compensate)
        position_world.append(position)
        prev_pos = cur_pos

    return position_world


def get_relative_coord(origin, points):
    """
    Get pose (x, y, yaw) relative to a given origin, yaw normalized to [-pi,+pi]

    The angle should be given in radians.
    :param points: vector of points
    :param origin: origin point
    :return: vector of points
    """
    ox, oy, oyaw = origin
    px, py, pyaw = np.array(points[0]), np.array(points[1]), np.array(points[2])

    # points coordinates relative to origin
    x_rel = np.cos(oyaw) * (px - ox) + np.sin(oyaw) * (py - oy)
    y_rel = -np.sin(oyaw) * (px - ox) + np.cos(oyaw) * (py - oy)
    yaw_rel = np.arctan2(np.sin(pyaw - oyaw), np.cos(pyaw - oyaw))  # limit to plus/minus pi

    x, y = get_global_coord([ox, oy, oyaw], [x_rel, y_rel])

    return x_rel, y_rel, yaw_rel, [x, y]


def get_global_coord(global_p, local_p):
    """

    :param origin:
    :param points:
    :return:
    """

    ox, oy, oyaw = global_p
    px, py  = np.array(local_p[0]), np.array(local_p[1])

    x_global = np.cos(oyaw) * px - np.sin(oyaw) * py + ox
    y_global = np.sin(oyaw) * px + np.cos(oyaw) * py + oy

    return x_global, y_global


def compute_mock(ego_groundtruth):
    """
    Convert to USK(ego) frame of reference
    1 - EML X, 2 - EML Y, 3 - EML YAW
    :param ego_groundtruth:
    :return:
    """
    # origin coord
    ox, oy, oyaw = ego_groundtruth[0][1], ego_groundtruth[0][2], ego_groundtruth[0][3]
    # get all the points from segment(segment represented by prediction horizon)
    x =   [ego_groundtruth[elem][1] for elem in range(len(ego_groundtruth))]
    y =   [ego_groundtruth[elem][2] for elem in range(len(ego_groundtruth))]
    yaw = [ego_groundtruth[elem][3] for elem in range(len(ego_groundtruth))]

    # compen_xy represent the global coordinates computed from relative coordinates
    x_rel, y_rel, yaw_rel, compen_xy = get_relative_coord([ox, oy, oyaw], [x, y, yaw])

    mp_mock = ego_groundtruth
    for idx in range(ego_groundtruth.__len__()):
        mp_mock[idx][1] = x_rel[idx]
        mp_mock[idx][2] = y_rel[idx]
        mp_mock[idx][3] = yaw_rel[idx]
        mp_mock[idx].append(0.0)  # dtrack - covered distance at node; unit [m] - not used at the moment

    return mp_mock, compen_xy