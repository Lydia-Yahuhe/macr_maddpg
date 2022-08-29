import copy

import numpy as np
import cv2
import scipy.stats as st
from rtree import index

from flightSim.load import routings
from flightSim.utils import *

# border = [109.3, 116, 29, 33.5]
border = [109, 120, 26, 34]
scale = 100
width, length = resolution(border, scale)
channel = 3
decimal = 1
radius = 5
fps = 8
interval = 30
fl_list = [i * 300.0 for i in range(20, 29)]  # 6300~8400
fl_list += [i * 300.0 + 200.0 for i in range(29, 41)]  # 8900~12200
segment_property = {'color': (255, 255, 255), 'thickness': 1}
border_property = {'color': (255, 191, 0), 'thickness': 2}


def build_rt_index_with_list(points):
    idx = index.Index(properties=index.Property(dimension=3))
    for i, point in enumerate(points):
        idx.insert(i, make_bbox(point))
    return idx


# ---------
# opencv
# ---------
def search_routing_in_a_area(vertices):
    segments = {}
    check_list = []
    for key, routing in routings.items():
        wpt_list = routing.waypointList

        in_poly_idx = []
        for i, wpt in enumerate(wpt_list):
            loc = wpt.location
            in_poly = pnpoly(vertices, [loc.lng, loc.lat])
            if in_poly:
                in_poly_idx.append(i)

        if len(in_poly_idx) <= 0:
            continue

        size = i + 1
        min_idx, max_idx = max(min(in_poly_idx) - 1, 0), min(size, max(in_poly_idx) + 2)
        # print(key, min_idx, max_idx, in_poly_idx, size, len(wpt_list))

        new_wpt_list = wpt_list[min_idx:max_idx]
        assert len(new_wpt_list) >= 2
        for i, wpt in enumerate(new_wpt_list[1:]):
            last_wpt = new_wpt_list[i]
            name_f, name_l = last_wpt.id + '-' + wpt.id, wpt.id + '-' + last_wpt.id

            if name_f not in check_list:
                segments[name_f] = [[last_wpt.location.lng, last_wpt.location.lat],
                                    [wpt.location.lng, wpt.location.lat]]
                check_list += [name_l, name_f]

    return segments


def generate_wuhan_base_map(frame_size, save_path=None, show=None):
    # 武汉空域边界点
    vertices = [(109.51666666666667, 31.9), (110.86666666666666, 33.53333333333333),
                (114.07, 32.125), (115.81333333333333, 32.90833333333333),
                (115.93333333333334, 30.083333333333332), (114.56666666666666, 29.033333333333335),
                (113.12, 29.383333333333333), (109.4, 29.516666666666666),
                (109.51666666666667, 31.9), (109.51666666666667, 31.9)]

    # 创建一个的黑色画布，RGB(0,0,0)即黑色
    image = np.zeros(frame_size, np.uint8)

    # 将空域边界画在黑色画布上
    points = convert_coord_to_pixel(vertices, border=border, scale=scale)
    points = np.array(points, np.int32).reshape((-1, 1, 2,))
    cv2.polylines(image, [points], True, border_property['color'], border_property['thickness'])

    # 将航路段画在黑色画布上
    segments = search_routing_in_a_area(vertices)
    for coord in segments.values():
        coord_idx = convert_coord_to_pixel(coord, border=border, scale=scale)
        cv2.line(image, coord_idx[0], coord_idx[1], segment_property['color'], segment_property['thickness'])

    if save_path is not None:
        cv2.imwrite(save_path, image)

    if show is not None:
        cv2.imshow("wuhan", image)
        cv2.waitKey(show)
        cv2.destroyAllWindows()

    return image


def add_points_on_base_map(points, image, font_scale=0.4, color=(255, 255, 255), font=cv2.FONT_HERSHEY_SIMPLEX,
                           **kwargs):
    points_just_coord = []
    for [name, lng, lat, alt, *point] in points:
        coord = [lng, lat]
        coord_idx = convert_coord_to_pixel([coord], border=border, scale=scale)[0]

        range_mixed = min(510, max((alt - 6000) / 4100 * 510, 0))
        if range_mixed <= 255:
            cv2.circle(image, coord_idx, radius, (0, 255, range_mixed), -1)
        else:
            cv2.circle(image, coord_idx, radius, (0, 510 - range_mixed, 255), -1)

        if name in kwargs['conflict_ac']:
            heading_spd_point = destination(coord, point[-1], 600 / 3600 * point[0] * NM2M)
            add_lines_on_base_map([[coord, heading_spd_point, False]], image, display=False)

            bbox_coords = get_bbox_2d(coord, ext=(0.5, 0.5))
            for i, pos in enumerate(bbox_coords[:-1]):
                add_lines_on_base_map([[pos, bbox_coords[i+1], False]], image, display=False)

            [x, y] = coord_idx

            cv2.putText(image, name, (x, y + 10), font, font_scale, color, 1)
            state = 'Altitude: {}'.format(round(alt, decimal))
            cv2.putText(image, state, (x, y + 30), font, font_scale, color, 1)
            state = '   Speed: {}({})'.format(round(point[0], decimal), round(point[1], decimal))
            cv2.putText(image, state, (x, y + 50), font, font_scale, color, 1)
            state = ' Heading: {}'.format(round(point[2], decimal))
            cv2.putText(image, state, (x, y + 70), font, font_scale, color, 1)

        points_just_coord.append((lng, lat, alt))
    return image, points_just_coord


def add_texts_on_base_map(texts, image, pos, color=(255, 255, 255), font_scale=0.4, font=cv2.FONT_HERSHEY_SIMPLEX):
    x, y = pos
    i = 0
    for key, text in texts.items():
        if isinstance(text, str):
            string = "{}: {}".format(key, text)
            cv2.putText(image, string, (x, y + i * 20), font, font_scale, color, 1)
            i += 1
        else:
            for j, text_ in enumerate(text):
                string = "{}_{}: {}".format(key, j + 1, text_)
                cv2.putText(image, string, (x, y + i * 20), font, font_scale, color, 1)
                i += 1
    return image


def add_lines_on_base_map(lines, image, color=(255, 0, 255), display=True, font_scale=0.4,
                          font=cv2.FONT_HERSHEY_SIMPLEX):
    if len(lines) <= 0:
        return image

    for [pos0, pos1, *other] in lines:
        if other[-1]:
            color = (255, 0, 255)

        [start, end] = convert_coord_to_pixel([pos0, pos1], border=border, scale=scale)
        cv2.line(image, start, end, color, 1)

        if display:
            [h_dist, v_dist] = other[:2]
            mid_idx = (int((start[0] + end[0]) / 2) + 10, int((start[1] + end[1]) / 2) + 10)
            state = ' H_dist: {}, V_dist: {}'.format(round(h_dist, decimal), round(v_dist, decimal))
            cv2.putText(image, state, mid_idx, font, font_scale, color, 1)
    return image


def add_scatter_on_base_map(x, y, pos, image, color=(255, 0, 255), font_scale=0.4, text=False, ver=20,
                            font=cv2.FONT_HERSHEY_SIMPLEX):
    start, end = pos
    cv2.line(image, pos, (start, end - int(y*ver)), color, 1)
    if text:
        cv2.putText(image, str(x//interval), (start, end + 10), font, font_scale, color, 1)
    return image


def add_hist_on_base_map(lst, pos, image, color=(255, 0, 255), font_scale=0.2, text=False, ver=20,
                         font=cv2.FONT_HERSHEY_SIMPLEX):
    if len(lst) <= 1:
        return

    lst = np.array(lst)  # 转化为1D array
    plot = np.array(fl_list)  # 对于单变量要使用1D array
    scipy_kde = st.gaussian_kde(lst)  # 高斯核密度估计
    density = scipy_kde.pdf(plot)  # pdf求概率密度

    start, end = pos
    for i, (x, y) in enumerate(zip(plot, density)):
        x_axis = start+i*10
        y_axis = max(end - int(y*ver), end-100)
        cv2.line(image, (x_axis, end), (x_axis, y_axis), color, 1)
        if text and i % 2 == 0:
            cv2.putText(image, str(x//100), (x_axis, end + 10), font, font_scale, color, 1)
    return image


def render(agents, conflicts, save_path, wait=1):
    print(width, length, channel)
    # print(fl_list)
    # 生成武汉空域底图
    # generate_wuhan_base_map((length, width, channel), save_path='scripts/wuhan_base.jpg', show=wait)

    # 武汉扇区的底图（有航路）
    base_img = cv2.imread('wuhan_base.jpg', cv2.IMREAD_COLOR)

    frame_size = (width, length)
    video_out = cv2.VideoWriter(save_path + '.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, frame_size)

    # 轨迹点
    points_dict = {}
    alt_dict = {}
    for key, agent in agents.items():
        for t, track in agent.tracks.items():
            track = [key] + track
            if t in points_dict.keys():
                points_dict[t].append(track)
                alt_dict[t].append(track[3])
            else:
                points_dict[t] = [track]
                alt_dict[t] = [track[3]]

    keys = list(points_dict.keys())
    conflict_info = {'>>> Conflict Information': ''}
    conflict_ac = {}
    x, y_conflict, y_flow, count = 0, 0, 0, 0
    for t in range(min(keys), max(keys) + 1):
        if x % interval == 0:
            count += 1
            pos = (20 + count*2, 770)
            base_img = add_scatter_on_base_map(x, y_flow, pos, base_img, ver=1, color=(0, 255, 225))
            base_img = add_scatter_on_base_map(x, y_conflict, pos, base_img, text=(count-1) % 20 == 0)
            y_conflict = 0

        image = copy.deepcopy(base_img)
        points = points_dict[t]

        # 全局信息
        global_info = {'>>> Information': '', 'Time': '{}, ac_en: {}, speed: x{}'.format(t, len(points), fps)}
        frame = add_texts_on_base_map(global_info, image, (700, 30), color=(255, 255, 0))

        # 冲突信息
        if t in conflicts.keys():
            strings, acs = [], []
            for c in conflicts[t]:
                strings.append(c.to_string())
                acs += c.id.split('-')
            conflict_info[t] = strings
            conflict_ac[t] = list(set(acs))
        else:
            strings = []
        x = t
        y_conflict += len(strings)
        y_flow = len(points)
        frame = add_hist_on_base_map(alt_dict[t], (700, 770), frame, ver=int(1e5), color=(0, 255, 225), text=True)

        if t - 300 in conflict_info.keys():
            conflict_info.pop(t - 300)
            conflict_ac.pop(t - 300)
        frame = add_texts_on_base_map(conflict_info, frame, (700, 80), color=(180, 238, 180))

        conflict_acs = []
        for acs in conflict_ac.values():
            conflict_acs += acs
        frame, points_just_coord = add_points_on_base_map(points, frame, conflict_ac=conflict_acs)

        idx = build_rt_index_with_list(points_just_coord)
        lines = []
        for i, [name, *point] in enumerate(points):
            if name not in conflict_acs:
                continue

            pos0 = point[:3]
            for j in idx.intersection(make_bbox(pos0, ext=(0.1, 0.1, 300))):
                if i == j:
                    continue

                pos1 = points_just_coord[j]
                h_dist = distance(pos0, pos1) / 1000
                v_dist = abs(pos0[-1] - pos1[-1])
                has_conflict = h_dist <= 10 and v_dist < 300
                lines.append([pos0, pos1, h_dist, v_dist, has_conflict])

        frame = add_lines_on_base_map(lines, image, color=(0, 0, 255))

        cv2.namedWindow(save_path, cv2.WINDOW_AUTOSIZE)
        cv2.imshow(save_path, frame)
        if cv2.waitKey(wait) == 113:
            video_out.release()
            cv2.waitKey(1) & 0xFF
            cv2.destroyAllWindows()
            return

        video_out.write(frame)

    video_out.release()
    cv2.waitKey(1) & 0xFF
    cv2.destroyAllWindows()
