import numpy as np
import cv2
from rtree import index

from flightSim.load import load_data_set
from flightSim.utils import *

border_origin = [109.3, 116, 29, 33.5]
border = [109, 120, 26, 34]
global_bbox = (border[0], border[2], 0.0, border[1], border[3], 12000.0)
scale = 100
width, length = resolution(border, scale)
channel = 3
decimal = 1
radius = 5
fps = 8
interval = 30
fl_list = [i * 300.0 for i in range(20, 29)]  # 6300~8400
fl_list += [i * 300.0 + 200.0 for i in range(29, 41)]  # 8900~12200
segment_property = {'color': (107, 55, 19), 'thickness': 1}  # BGR
border_property = {'color': (100, 100, 100), 'thickness': 2}


def build_rt_index_with_list(points):
    idx = index.Index(properties=index.Property(dimension=3))
    for i, point in enumerate(points):
        idx.insert(i, make_bbox(point))
    return idx


# ---------
# opencv
# ---------
def search_routing_in_a_area(vertices):
    route_dict = load_data_set().routings

    segments = {}
    check_list = []
    for key, routing in route_dict.items():
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
    image = np.ones(frame_size, np.uint8) * 255

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


def add_points_on_base_map(points, image, font_scale=0.4, color=(0, 0, 0), font=cv2.FONT_HERSHEY_SIMPLEX,
                           **kwargs):
    count = 0
    for [name, lng, lat, alt, *point] in points:
        coord = [lng, lat]
        coord_idx = convert_coord_to_pixel([coord], border=border, scale=scale)[0]

        # 如果飞机是参与冲突的
        if name in kwargs['conflict_ac']:
            # 每个飞机都是个圆，圆的颜色代表飞行高度，Green（低）-Yellow-Red（高），
            range_mixed = min(510, max((alt - 6000) / 4100 * 510, 0))
            if range_mixed <= 255:
                cv2.circle(image, coord_idx, radius, (0, 255, range_mixed), -1)
            else:
                cv2.circle(image, coord_idx, radius, (0, 510 - range_mixed, 255), -1)

            # 紫色直线代表其航向，长度代表速度
            heading_spd_point = destination(coord, point[-1], 600 / 3600 * point[0] * NM2M)
            add_lines_on_base_map([[coord, heading_spd_point, False]], image,
                                  color=(255, 0, 0), display=False, thickness=2)

            # 画观察范围
            bbox_coords = get_bbox_2d(coord, ext=(1.0, 1.0))
            for i, pos in enumerate(bbox_coords[:-1]):
                add_lines_on_base_map([[pos, bbox_coords[i+1], False]], image, display=False)

            # 加上呼号
            cv2.putText(image, name, (coord_idx[0], coord_idx[1] + 10), font, font_scale, color, 1)

            cmd_dict = kwargs['ac_cmd_dict'][name]
            x, y = 750, 340+count*80
            cv2.putText(image, name, (x, y), font, font_scale, color, 1)
            state = 'Alt: {}({})'.format(round(alt, decimal), cmd_dict['ALT'])
            cv2.putText(image, state, (x+20, y + 20), font, font_scale, color, 1)
            state = 'Spd: {}({})({})'.format(round(point[0], decimal), round(point[1], decimal), cmd_dict['SPD'])
            cv2.putText(image, state, (x+20, y + 40), font, font_scale, color, 1)
            state = 'Hdg: {}({})'.format(round(point[2], decimal), cmd_dict['HDG'])
            cv2.putText(image, state, (x+20, y + 60), font, font_scale, color, 1)
            count += 1
        else:
            cv2.circle(image, coord_idx, int(radius*0.6), (87, 139, 46), -1)

    return image


def add_texts_on_base_map(texts, image, pos, color=(255, 255, 255), font_scale=0.4, font=cv2.FONT_HERSHEY_SIMPLEX,
                          thickness=1):
    x, y = pos
    i = 0
    for key, text in texts.items():
        if isinstance(text, str):
            string = "{}: {}".format(key, text)
            cv2.putText(image, string, (x, y + i * 20), font, font_scale, color, thickness)
            i += 1
        else:
            for j, text_ in enumerate(text):
                string = "{}_{}: {}".format(key, j + 1, text_)
                cv2.putText(image, string, (x, y + i * 20), font, font_scale, color, thickness)
                i += 1
    return image


def add_lines_on_base_map(lines, image, color=(255, 0, 255), display=True, font_scale=0.4, thickness=1,
                          font=cv2.FONT_HERSHEY_SIMPLEX):
    if len(lines) <= 0:
        return image

    for [pos0, pos1, *other] in lines:
        if len(other) > 0 and other[-1]:
            color = (255, 0, 255)

        [start, end] = convert_coord_to_pixel([pos0, pos1], border=border, scale=scale)
        cv2.line(image, start, end, color, thickness)

        if display:
            [h_dist, v_dist] = other[:2]
            mid_idx = (int((start[0] + end[0]) / 2) + 10, int((start[1] + end[1]) / 2) + 10)
            state = ' H_dist: {}, V_dist: {}'.format(round(h_dist, decimal), round(v_dist, decimal))
            cv2.putText(image, state, mid_idx, font, font_scale, color, 1)
    return image


# generate_wuhan_base_map((length, width, channel), save_path='../scripts/wuhan_base.jpg', show=1000)
