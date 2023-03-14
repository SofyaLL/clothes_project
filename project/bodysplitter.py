import typing as t
import math
import glm
import enum
import numpy as np
import cv2
import mediapipe as mp

BODY_CROPS_NAME = ['head', 'hips', 'body', 'body_and_hips', 'right_foot', 'left_foot']


class Point(t.NamedTuple):
    x: float
    y: float


class Line(t.NamedTuple):
    A: Point
    B: Point


class Rect(t.NamedTuple):
    A: Point
    B: Point
    C: Point
    D: Point


class Geometry(t.NamedTuple):
    rect_head: Rect
    rect_hips: Rect
    rect_body: Rect
    rect_body_and_hips: Rect
    rect_right_foot: Rect
    rect_left_foot: Rect


class BodyCrops(t.NamedTuple):
    head: t.Optional[np.ndarray]
    hips: t.Optional[np.ndarray]
    body: t.Optional[np.ndarray]
    body_and_hips: t.Optional[np.ndarray]
    right_foot: t.Optional[np.ndarray]
    left_foot: t.Optional[np.ndarray]


class BlazePoseBase:
    nose: int = 0
    left_eye_inner: int = 1
    left_eye: int = 2
    left_eye_outer: int = 3
    right_eye_inner: int = 4
    right_eye: int = 5
    right_eye_outer: int = 6
    left_ear: int = 7
    right_ear: int = 8
    mouth_left: int = 9
    mouth_right: int = 10
    left_shoulder: int = 11
    right_shoulder: int = 12
    left_elbow: int = 13
    right_elbow: int = 14
    left_wrist: int = 15
    right_wrist: int = 16
    left_pinky: int = 17
    right_pinky: int = 18
    left_index: int = 19
    right_index: int = 20
    left_thumb: int = 21
    right_thumb: int = 22
    left_hip: int = 23
    right_hip: int = 24
    left_knee: int = 25
    right_knee: int = 26
    left_ankle: int = 27
    right_ankle: int = 28
    left_heel: int = 29
    right_heel: int = 30
    left_foot_index: int = 31
    right_foot_index: int = 32
    total_count: int = 33


class BlazePose(enum.Enum):
    nose: int = 0
    left_eye_inner: int = 1
    left_eye: int = 2
    left_eye_outer: int = 3
    right_eye_inner: int = 4
    right_eye: int = 5
    right_eye_outer: int = 6
    left_ear: int = 7
    right_ear: int = 8
    mouth_left: int = 9
    mouth_right: int = 10
    left_shoulder: int = 11
    right_shoulder: int = 12
    left_elbow: int = 13
    right_elbow: int = 14
    left_wrist: int = 15
    right_wrist: int = 16
    left_pinky: int = 17
    right_pinky: int = 18
    left_index: int = 19
    right_index: int = 20
    left_thumb: int = 21
    right_thumb: int = 22
    left_hip: int = 23
    right_hip: int = 24
    left_knee: int = 25
    right_knee: int = 26
    left_ankle: int = 27
    right_ankle: int = 28
    left_heel: int = 29
    right_heel: int = 30
    left_foot_index: int = 31
    right_foot_index: int = 32
    total_count: int = 33


def spoint(point: t.Any) -> Point:
    return Point(point.x, point.y)


def get_middle_point(l: t.Optional[Line] = None, p1: t.Optional[Point] = None, p2: t.Optional[Point] = None) -> Point:
    if p1 is None and p2 is None:
        return Point(min(l.A.x, l.B.x) + abs(l.A.x - l.B.x) / 2, min(l.A.y, l.B.y) + abs(l.A.y - l.B.y) / 2)
    else:
        l = Line(Point(p1.x, p1.y), Point(p2.x, p2.y))
        return Point(min(l.A.x, l.B.x) + abs(l.A.x - l.B.x) / 2, min(l.A.y, l.B.y) + abs(l.A.y - l.B.y) / 2)


def get_line_length(l: Line) -> float:
    return math.sqrt((l.A.x - l.B.x) * (l.A.x - l.B.x) + (l.A.y - l.B.y) * (l.A.y - l.B.y))


def get_line_extension(l: Line, extension: float) -> Point:
    length: float = get_line_length(l)
    try:
        dx_add: float = (l.A.x - l.B.x) * extension / length
        dy_add: float = (l.A.y - l.B.y) * extension / length
    except ZeroDivisionError:
        dx_add, dy_add = 0, 0
    return Point(l.A.x + dx_add, l.A.y + dy_add)


def get_rect_extension(r: Rect, extension: float):
    A = get_line_extension(Line(r.A, r.D), extension / 2)
    B = get_line_extension(Line(r.B, r.C), extension / 2)
    A = get_line_extension(Line(A, B), extension / 2)
    B = get_line_extension(Line(B, A), extension / 2)
    C = get_line_extension(Line(r.C, r.B), extension / 2)
    D = get_line_extension(Line(r.D, r.A), extension / 2)
    C = get_line_extension(Line(C, D), extension / 2)
    D = get_line_extension(Line(D, C), extension / 2)
    extended: Rect = Rect(A, B, C, D)

    return extended


def get_distance_from_point_to_line(p: Point, l: Line) -> float:
    normal_length = get_line_length(l)
    try:
        distance = ((p.x - l.A.x) * (l.B.y - l.A.y) - (p.y - l.A.y) * (l.B.x - l.A.x)) / normal_length
    except ZeroDivisionError:
        return 0
    return distance


def get_head_rect(r_eye: Point, l_eye: Point, r_mouth: Point, l_mouth: Point, nose: Point) -> Rect:
    HTNBM_Rate = 1.8
    NBMHB_Rate = 0.8

    eye_c: Point = get_middle_point(p1=r_eye, p2=l_eye)
    mouth_c: Point = get_middle_point(p1=r_mouth, p2=l_mouth)

    face_base_len = get_line_length(Line(eye_c, mouth_c))

    face_top: Point = get_line_extension(Line(eye_c, mouth_c), HTNBM_Rate * face_base_len)
    face_bot: Point = get_line_extension(Line(mouth_c, eye_c), NBMHB_Rate * face_base_len)

    nose_deviation = get_distance_from_point_to_line(nose, Line(face_bot, face_top))

    move_coef: float = math.fabs(nose_deviation) / get_line_length(Line(eye_c, mouth_c))

    dx: float = face_bot.x - face_top.x
    dy: float = face_bot.y - face_top.y

    topLeftX: float = face_top.x - dy * (1 - move_coef) / 2 if nose_deviation >= 0 else face_top.x - dy * (1 + move_coef) / 2
    topLeftY: float = face_top.y + dx * (1 - move_coef) / 2 if nose_deviation >= 0 else face_top.y + dx * (1 + move_coef) / 2

    topRigxtX: float = face_top.x + dy * (1 + move_coef) / 2 if nose_deviation >= 0 else face_top.x + dy * (1 - move_coef) / 2
    topRightY: float = face_top.y - dx * (1 + move_coef) / 2 if nose_deviation >= 0 else face_top.y - dx * (1 - move_coef) / 2

    botRigxtX: float = face_bot.x + dy * (1 + move_coef) / 2 if nose_deviation >= 0 else face_bot.x + dy * (1 - move_coef) / 2
    botRightY: float = face_bot.y - dx * (1 + move_coef) / 2 if nose_deviation >= 0 else face_bot.y - dx * (1 - move_coef) / 2

    botLeftX = face_bot.x - dy * (1 - move_coef) / 2 if nose_deviation >= 0 else face_bot.x - dy * (1 + move_coef) / 2
    botLeftY = face_bot.y + dx * (1 - move_coef) / 2 if nose_deviation >= 0 else face_bot.y + dx * (1 + move_coef) / 2

    return Rect(Point(topLeftX, topLeftY), Point(topRigxtX, topRightY), Point(botRigxtX, botRightY), Point(botLeftX, botLeftY))



def get_leg_rect(hip: Point, knee: Point, ankle: Point, heel: Point, big_finger: Point):
        FKA_Rate = 0.4
        width_1 = get_line_length(Line(hip, knee)) * FKA_Rate
        width_2 = get_line_length(Line(knee, ankle)) * FKA_Rate
        width_3 = get_line_length(Line(heel, big_finger))

        width = max(max(width_1, width_2), width_3)

        knee_dist = get_distance_from_point_to_line(knee, Line(hip, heel))
        leg_base_length = get_line_length(Line(hip, heel))

        dx = knee_dist * (hip.y - heel.y) / leg_base_length
        dy = knee_dist * (hip.x - heel.x) / leg_base_length

        return get_rect_extension(Rect(Point(hip.x + dx, hip.y - dy), hip, ankle, Point(ankle.x + dx, ankle.y - dy)), width * 1.5)


def get_hand_rect(shoulder: Point, elbow: Point, wrist: Point, t_finger: Point, i_finger: Point, l_finger: Point):
        SEW_Rate = 0.3
        width_1 = get_line_length(Line(shoulder, elbow)) * SEW_Rate
        width_2 = get_line_length(Line(elbow, wrist)) * SEW_Rate
        width_3 = get_line_length(Line(wrist, t_finger))
        width_4 = get_line_length(Line(wrist, i_finger))
        width_5 = get_line_length(Line(wrist, l_finger))

        w = (width_1, width_2, width_3, width_4, width_5)
        width = max(w)

        elbow_dist = get_distance_from_point_to_line(elbow, Line(shoulder, wrist))

        hand_base_length = get_line_length(Line(shoulder, wrist ))

        dx = elbow_dist * (wrist.y - shoulder.y) / hand_base_length
        dy = elbow_dist * (wrist.x - shoulder.x) / hand_base_length

        return get_rect_extension(Rect(Point(shoulder.x + dx, shoulder.y - dy), shoulder, wrist, Point(wrist.x + dx, wrist.y - dy)), width * 1.5)


def getTorsoRect(r_startpoint: Point, l_startpoint: Point, r_hip: Point, l_hip: Point):
        shoulder_center: Point = get_middle_point(p1=r_startpoint, p2=l_startpoint)
        hip_center: Point = get_middle_point(p1=r_hip, p2=l_hip)

        add_to_top = math.fabs((r_startpoint.y - shoulder_center.y) / (r_startpoint.x - shoulder_center.x) * get_line_length(Line(r_startpoint, l_startpoint)) / 2)
        add_to_bot = math.fabs((r_hip.y - hip_center.y) / (r_hip.x - hip_center.x) * get_line_length(Line(r_hip, l_hip)) / 2)

        base_top: Point = get_line_extension(Line(shoulder_center , hip_center), add_to_top)
        base_bot: Point = get_line_extension(Line(hip_center , shoulder_center), add_to_bot)

        torso_base_length: float = get_line_length(Line(base_top, base_bot))

        torso_width: float = max(get_line_length(Line(r_startpoint, l_startpoint)), get_line_length(Line(r_hip, l_hip)))

        dx: float = torso_width * (base_top.y - base_bot.y) / (2 * torso_base_length)
        dy: float = torso_width * (base_top.x - base_bot.x) / (2 * torso_base_length)

        return get_rect_extension(Rect(Point(base_top.x + dx, base_top.y - dy), Point(base_top.x - dx, base_top.y + dy),
                                    Point(base_bot.x - dx, base_bot.y + dy), Point(base_bot.x + dx, base_bot.y - dy )), torso_base_length / 4)


def getHipsRect(r_hip: Point, l_hip: Point, r_endpoint: Point, l_endpoint: Point):
        hip_center = get_middle_point(p1=r_hip, p2=l_hip)
        knee_center = get_middle_point(p1=r_endpoint, p2=l_endpoint)

        add_to_top = math.fabs((r_hip.y - hip_center.y) / (r_hip.x - hip_center.x) * get_line_length(Line( r_hip, l_hip)) / 2)
        add_to_bot = math.fabs((r_endpoint.y - knee_center.y) / (r_endpoint.x - knee_center.x) * get_line_length(Line(r_endpoint, l_endpoint)) / 2)

        base_top: Point = get_line_extension(Line(hip_center , knee_center), add_to_top)
        base_bot: Point = get_line_extension(Line(knee_center , hip_center), add_to_bot)

        hip_base_length: float = get_line_length(Line(base_top, base_bot))

        hip_width: float = max(get_line_length(Line(r_hip, l_hip)), get_line_length(Line(r_endpoint, l_endpoint)))

        dx: float = hip_width * (base_top.y - base_bot.y) / (2 * hip_base_length)
        dy: float = hip_width * (base_top.x - base_bot.x) / (2 * hip_base_length)

        return get_rect_extension(Rect(Point(base_top.x + dx, base_top.y - dy), Point(base_top.x - dx, base_top.y + dy),
                                    Point(base_bot.x - dx, base_bot.y + dy), Point(base_bot.x + dx, base_bot.y - dy )), hip_base_length / 3)


def getTorsoAndHipsRect(r_shoulder: Point, l_shoulder: Point, r_hip: Point, l_hip: Point, r_knee: Point, l_knee: Point):
        shoulder_center: Point = get_middle_point(p1=r_shoulder, p2=l_shoulder)
        knee_center: Point = get_middle_point(p1=r_knee, p2=l_knee)

        add_to_top: float = math.fabs((r_shoulder.y - shoulder_center.y) / (r_shoulder.x - shoulder_center.x) * get_line_length(Line(r_shoulder, l_shoulder)) / 2)
        add_to_bot: float = math.fabs((r_knee.y - knee_center.y) / (r_knee.x - knee_center.x) * get_line_length(Line(r_knee, l_knee)) / 2)

        base_top: Point = get_line_extension(Line(shoulder_center , knee_center), add_to_top)
        base_bot: Point = get_line_extension(Line(knee_center , shoulder_center), add_to_bot)

        points = [r_shoulder, l_shoulder, r_hip, l_hip, r_knee, l_knee]
        distances = []
        for p in points:
            distances.append(get_distance_from_point_to_line(p, Line(base_top, base_bot)))

        max_pos_dev: float = max(distances)
        max_neg_dev: float = min(distances)

        base_length: float = get_line_length(Line(base_top, base_bot))

        dx_pos: float = math.fabs(max_pos_dev * (base_top.y - base_bot.y) / base_length)
        dy_pos: float = max_pos_dev * (base_top.x - base_bot.x) / base_length

        dx_neg: float = math.fabs(max_neg_dev * (base_top.y - base_bot.y) / base_length)
        dy_neg: float = max_neg_dev * (base_top.x - base_bot.x) / base_length

        return get_rect_extension(Rect(Point(base_top.x - dx_neg, base_top.y + dy_neg), Point(base_top.x + dx_pos, base_top.y + dy_pos),
            Point(base_bot.x + dx_pos, base_bot.y + dy_pos), Point(base_bot.x - dx_neg, base_bot.y + dy_neg)),
            (math.fabs(max_pos_dev) + math.fabs(max_neg_dev)) / 3)

def getFootRect(ankle: Point, heel: Point, big_finger: Point) -> Rect:
        af: float = get_line_length(Line(ankle, big_finger))
        hf: float = get_line_length(Line(heel, big_finger))
        ad: float = get_distance_from_point_to_line(ankle, Line(heel, big_finger))

        foot_base_length: float = get_line_length(Line(heel, big_finger))

        foot_width: float = foot_base_length if foot_base_length > math.fabs(ad) else math.fabs(ad)

        try:
            dx: float = ad * (heel.y - big_finger.y) / foot_base_length
            dy: float = ad * (heel.x - big_finger.x) / foot_base_length
        except ZeroDivisionError:
            dx, dy = 0, 0

        if af < hf:
            return get_rect_extension(Rect(Point( heel.x - dx, heel.y + dy), heel, big_finger, Point(big_finger.x - dx, big_finger.y + dy)), foot_width)
        else:
            return get_rect_extension(Rect(Point(ankle.x + dx, ankle.y - dy), ankle, Point(big_finger.x - dx, big_finger.y + dy), big_finger), foot_width)


def getWristRect(wrist: Point, t_finger: Point, i_finger: Point, l_finger: Point) -> Rect:
        width_1: float = get_line_length(Line(wrist, t_finger))
        width_2: float = get_line_length(Line(wrist, i_finger))
        width_3: float = get_line_length(Line(wrist, l_finger))

        w = [width_1, width_2, width_3]
        width: float = max(w)

        wrist_base_line = Line(wrist, get_middle_point(p1=i_finger, p2=l_finger))

        dx: float = (wrist_base_line.A.y - wrist_base_line.B.y) / 2
        dy: float = (wrist_base_line.A.x - wrist_base_line.B.x) / 2

        C: Point = get_middle_point(wrist_base_line)
        D: Point = C
        C = Point(C.x + dx, C.y - dy)
        D = Point(D.x - dx, D.y + dy)

        r: Rect = Rect(wrist_base_line.A , C, wrist_base_line.B, D)

        wrist_base_rect: Rect = Rect(get_middle_point(p1=r.A, p2=r.B) , get_middle_point(p1=r.B, p2=r.C), get_middle_point(p1=r.C, p2=r.D), get_middle_point(p1=r.D, p2=r.A))

        return get_rect_extension(wrist_base_rect, width*2)


def blaze_all(landmarks: t.List[glm.vec3]) -> dict:
    result: dict = {}

    result['nose'] = {'x': spoint(landmarks[BlazePoseBase.nose]).x, 'y': spoint(landmarks[BlazePoseBase.nose]).y, 'p': 0.9}
    result['left_eye'] = {'x': spoint(landmarks[BlazePoseBase.left_eye]).x, 'y': spoint(landmarks[BlazePoseBase.left_eye]).y, 'p': 0.9}
    result['right_eye'] = {'x': spoint(landmarks[BlazePoseBase.right_eye]).x, 'y': spoint(landmarks[BlazePoseBase.right_eye]).y, 'p': 0.9}
    result['left_ear'] = {'x': spoint(landmarks[BlazePoseBase.left_ear]).x, 'y': spoint(landmarks[BlazePoseBase.left_ear]).y, 'p': 0.9}
    result['right_ear'] = {'x': spoint(landmarks[BlazePoseBase.right_ear]).x, 'y': spoint(landmarks[BlazePoseBase.right_ear]).y, 'p': 0.9}
    result['left_shoulder'] = {'x': spoint(landmarks[BlazePoseBase.left_shoulder]).x, 'y': spoint(landmarks[BlazePoseBase.left_shoulder]).y, 'p': 0.9}
    result['right_shoulder'] = {'x': spoint(landmarks[BlazePoseBase.right_shoulder]).x, 'y': spoint(landmarks[BlazePoseBase.right_shoulder]).y, 'p': 0.9}
    result['left_elbow'] = {'x': spoint(landmarks[BlazePoseBase.left_elbow]).x, 'y': spoint(landmarks[BlazePoseBase.left_elbow]).y, 'p': 0.9}
    result['right_elbow'] = {'x': spoint(landmarks[BlazePoseBase.right_ear]).x, 'y': spoint(landmarks[BlazePoseBase.right_elbow]).y, 'p': 0.9}
    result['left_wrist'] = {'x': spoint(landmarks[BlazePoseBase.left_wrist]).x, 'y': spoint(landmarks[BlazePoseBase.left_wrist]).y, 'p': 0.9}
    result['right_wrist'] = {'x': spoint(landmarks[BlazePoseBase.right_wrist]).x, 'y': spoint(landmarks[BlazePoseBase.right_wrist]).y, 'p': 0.9}
    result['left_hip'] = {'x': spoint(landmarks[BlazePoseBase.left_hip]).x, 'y': spoint(landmarks[BlazePoseBase.left_hip]).y, 'p': 0.9}
    result['right_hip'] = {'x': spoint(landmarks[BlazePoseBase.right_hip]).x, 'y': spoint(landmarks[BlazePoseBase.right_hip]).y, 'p': 0.9}
    result['left_knee'] = {'x': spoint(landmarks[BlazePoseBase.left_knee]).x, 'y': spoint(landmarks[BlazePoseBase.left_knee]).y, 'p': 0.9}
    result['right_knee'] = {'x': spoint(landmarks[BlazePoseBase.right_knee]).x, 'y': spoint(landmarks[BlazePoseBase.right_knee]).y, 'p': 0.9}
    result['left_ankle'] = {'x': spoint(landmarks[BlazePoseBase.left_ankle]).x, 'y': spoint(landmarks[BlazePoseBase.left_ankle]).y, 'p': 0.9}
    result['right_ankle'] = {'x': spoint(landmarks[BlazePoseBase.right_ankle]).x, 'y': spoint(landmarks[BlazePoseBase.right_ankle]).y, 'p': 0.9}
    result['neck'] = {'x': spoint(landmarks[BlazePoseBase.right_eye]).x + spoint(landmarks[BlazePoseBase.right_eye_outer]).x / 2,
                      'y': spoint(landmarks[BlazePoseBase.right_eye]).y + spoint(landmarks[BlazePoseBase.right_eye_outer]).y / 2, 'p': 0.9}

    return result


def blaze_head(landmarks: t.List[glm.vec3]) -> Rect:
    r_eye = spoint(landmarks[BlazePoseBase.right_eye])
    l_eye = spoint(landmarks[BlazePoseBase.left_eye])
    r_mouth = spoint(landmarks[BlazePoseBase.mouth_right])
    l_mouth = spoint(landmarks[BlazePoseBase.mouth_left])
    nose = spoint(landmarks[BlazePoseBase.nose])

    return get_head_rect(r_eye, l_eye, r_mouth, l_mouth, nose)


def blaze_right_hand(landmarks: t.List[glm.vec3]) -> Rect:
    shoulder = spoint(landmarks[BlazePoseBase.right_shoulder])
    elbow = spoint(landmarks[BlazePoseBase.right_elbow])
    wrist = spoint(landmarks[BlazePoseBase.right_wrist])
    t_finger = spoint(landmarks[BlazePoseBase.right_pinky])
    i_finger = spoint(landmarks[BlazePoseBase.right_index])
    l_finger = spoint(landmarks[BlazePoseBase.right_thumb])

    return get_hand_rect(shoulder, elbow, wrist, t_finger, i_finger, l_finger)


def blaze_left_hand(landmarks: t.List[glm.vec3]) -> Rect:
    shoulder = spoint(landmarks[BlazePoseBase.left_shoulder])
    elbow = spoint(landmarks[BlazePoseBase.left_elbow])
    wrist = spoint(landmarks[BlazePoseBase.left_wrist])
    t_finger = spoint(landmarks[BlazePoseBase.left_pinky])
    i_finger = spoint(landmarks[BlazePoseBase.left_index])
    l_finger = spoint(landmarks[BlazePoseBase.left_thumb])

    return get_hand_rect(shoulder, elbow, wrist, t_finger, i_finger, l_finger)


def blaze_right_leg(landmarks: t.List[glm.vec3]):
    hip = spoint(landmarks[BlazePoseBase.right_hip])
    knee = spoint(landmarks[BlazePoseBase.right_knee])
    ankle = spoint(landmarks[BlazePoseBase.right_ankle])
    heel = spoint(landmarks[BlazePoseBase.right_heel])
    big_finger = spoint(landmarks[BlazePoseBase.right_foot_index])

    return get_leg_rect(hip, knee, ankle, heel, big_finger)


def blaze_left_leg(landmarks: t.List[glm.vec3]):
    hip = spoint(landmarks[BlazePoseBase.left_hip])
    knee = spoint(landmarks[BlazePoseBase.left_knee])
    ankle = spoint(landmarks[BlazePoseBase.left_ankle])
    heel = spoint(landmarks[BlazePoseBase.left_heel])
    big_finger = spoint(landmarks[BlazePoseBase.left_foot_index])

    return get_leg_rect(hip, knee, ankle, heel, big_finger)


def blaze_torso(landmarks: t.List[glm.vec3]):
    r_shoulder = spoint(landmarks[BlazePoseBase.right_shoulder])
    l_shoulder = spoint(landmarks[BlazePoseBase.left_shoulder])
    r_hip = spoint(landmarks[BlazePoseBase.right_hip])
    l_hip = spoint(landmarks[BlazePoseBase.left_hip])

    r_wrist = spoint(landmarks[BlazePoseBase.right_wrist])
    l_wrist = spoint(landmarks[BlazePoseBase.left_wrist])

    if l_wrist.y < l_shoulder.y:
        l_startpoint = l_wrist
    else:
        l_startpoint = l_shoulder

    if r_wrist.y < r_shoulder.y:
        r_startpoint = r_wrist
    else:
        r_startpoint = r_shoulder

    return getTorsoRect(r_startpoint, l_startpoint, r_hip, l_hip)


def blaze_hips(landmarks: t.List[glm.vec3]):
    r_hip = spoint(landmarks[BlazePoseBase.right_hip])
    l_hip = spoint(landmarks[BlazePoseBase.left_hip])
    r_knee = spoint(landmarks[BlazePoseBase.right_knee])
    l_knee = spoint(landmarks[BlazePoseBase.left_knee])

    l_heel = spoint(landmarks[BlazePoseBase.left_heel])
    r_heel = spoint(landmarks[BlazePoseBase.right_heel])

    return getHipsRect(r_hip, l_hip, r_heel, l_heel)


def blaze_torso_and_hips(landmarks: t.List[glm.vec3]):
    r_shoulder = spoint(landmarks[BlazePoseBase.right_shoulder])
    l_shoulder = spoint(landmarks[BlazePoseBase.left_shoulder])
    r_hip = spoint(landmarks[BlazePoseBase.right_hip])
    l_hip = spoint(landmarks[BlazePoseBase.left_hip])
    r_knee = spoint(landmarks[BlazePoseBase.right_knee])
    l_knee = spoint(landmarks[BlazePoseBase.left_knee])

    return getTorsoAndHipsRect(r_shoulder, l_shoulder, r_hip, l_hip, r_knee, l_knee)


def blaze_left_foot(landmarks: t.List[glm.vec3]):
    ankle = spoint(landmarks[BlazePoseBase.left_ankle])
    heel = spoint(landmarks[BlazePoseBase.left_heel])
    big_finger = spoint(landmarks[BlazePoseBase.left_foot_index])
    return getFootRect(ankle, heel, big_finger)


def blaze_right_foot(landmarks: t.List[glm.vec3]):
    ankle = spoint(landmarks[BlazePoseBase.right_ankle])
    heel = spoint(landmarks[BlazePoseBase.right_heel])
    big_finger = spoint(landmarks[BlazePoseBase.right_foot_index])
    return getFootRect(ankle, heel, big_finger)


def blaze_left_wrist(landmarks: t.List[glm.vec3]):
    wrist = spoint(landmarks[BlazePoseBase.left_wrist])
    t_finger = spoint(landmarks[BlazePoseBase.left_pinky])
    i_finger = spoint(landmarks[BlazePoseBase.left_index])
    l_finger = spoint(landmarks[BlazePoseBase.left_thumb])
    return getWristRect(wrist, t_finger, i_finger, l_finger)


def blaze_right_wrist(landmarks: t.List[glm.vec3]):
    wrist = spoint(landmarks[BlazePoseBase.right_wrist])
    t_finger = spoint(landmarks[BlazePoseBase.right_pinky])
    i_finger = spoint(landmarks[BlazePoseBase.right_index])
    l_finger = spoint(landmarks[BlazePoseBase.right_thumb])
    return getWristRect(wrist, t_finger, i_finger, l_finger)


def denormalized_point(image: np.ndarray, point: Point) -> Point:
    return Point(point.x * image.shape[1], point.y * image.shape[0])


def denormalized_rect(image: np.ndarray, rect: Rect) -> t.Tuple:
    a = denormalized_point(image, rect.A)
    b = denormalized_point(image, rect.B)
    c = denormalized_point(image, rect.C)
    d = denormalized_point(image, rect.D)

    return cv2.minAreaRect(np.array([list(a), list(b), list(c), list(d)], dtype=np.float32))


def normalized_rect(rect: Rect):
    a = list(Point(rect.A.x, rect.A.y))
    b = list(Point(rect.B.x, rect.B.y))
    c = list(Point(rect.C.x, rect.C.y))
    d = list(Point(rect.D.x, rect.D.y))

    return cv2.minAreaRect(np.array([a, b, c, d], dtype=np.float32))


def annotate_point(image: np.ndarray, point: Point, color) -> np.ndarray:
    return cv2.circle(image, denormalized_point(image, point), 2, color, 1)


def annotate_line(image: np.ndarray, p0: Point, p1: Point, color) -> np.ndarray:
    return cv2.line(image, denormalized_point(image, p0), denormalized_point(image, p1), color, 2, 1)


def annotate_rect(image: np.ndarray, rect: Rect, color) -> np.ndarray:
    image = annotate_point(image, rect.A, color)
    image = annotate_point(image, rect.B, color)
    image = annotate_point(image, rect.C, color)
    image = annotate_point(image, rect.D, color)

    image = annotate_line(image, rect.A, rect.B, color)
    image = annotate_line(image, rect.B, rect.C, color)
    image = annotate_line(image, rect.C, rect.D, color)
    image = annotate_line(image, rect.D, rect.A, color)

    return image


def simple_crop_rect(img: np.ndarray, rect: t.Tuple, use_rotation=True) -> np.ndarray:
    # rotate img
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    (height, width, _) = img.shape

    W = rect[1][0]
    H = rect[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    left, top, right, bottom = 0, 0, 0, 0
    if x1 < 0:
        left = abs(x1)
    elif x1 > width:
        left = width - 1
    if y1 < 0:
        top = abs(y1)
    elif y1 > height:
        top = height - 1
    if x2 < 0:
        right = abs(x2)
    elif x2 > width:
        right = width - 1
    if y2 < 0:
        bottom = abs(y2)
    elif y2 > height:
        bottom = height - 1

    img = cv2.copyMakeBorder(img, top=top, bottom=bottom, right=right, left=left, borderType=cv2.BORDER_CONSTANT)

    angle = rect[2]
    if angle < -45:
        angle += 90

    # Center of rectangle in source image
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    # Size of the upright rectangle bounding the rotated rectangle
    size = (x2 - x1, y2 - y1)
    M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)
    # Cropped upright rectangle
    cropped = cv2.getRectSubPix(img, size, center)
    if use_rotation:
        cropped = cv2.warpAffine(cropped, M, size, borderMode=cv2.BORDER_CONSTANT)
    croppedW = H if H > W else W
    croppedH = H if H < W else W
    # Final cropped & rotated rectangle
    # croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW), int(croppedH)), (size[0] / 2, size[1] / 2))
    return cropped


def fill_images(source: np.ndarray, resize_to_square: float, geometry: Geometry):
    head = simple_crop_rect(source, denormalized_rect(source, geometry.rect_head), use_rotation=False)
    hips = simple_crop_rect(source, denormalized_rect(source, geometry.rect_hips), use_rotation=False)
    body = simple_crop_rect(source, denormalized_rect(source, geometry.rect_body), use_rotation=False)
    body_and_hips = simple_crop_rect(source, denormalized_rect(source, geometry.rect_body_and_hips), use_rotation=False)
    right_foot = simple_crop_rect(source, denormalized_rect(source, geometry.rect_right_foot), use_rotation=False)
    left_foot = simple_crop_rect(source, denormalized_rect(source, geometry.rect_left_foot), use_rotation=False)

    return BodyCrops(head=head, hips=hips, body=body, body_and_hips=body_and_hips, right_foot=right_foot,
                     left_foot=left_foot)


def get_person(geometry: Geometry):
    body = geometry.rect_body
    head = geometry.rect_head
    l_foot = geometry.rect_left_foot
    r_foot = geometry.rect_right_foot
    top = head.A.y
    left = body.A.x
    right = body.B.x
    bottom = max(l_foot.D.y, r_foot.D.y)
    return left, top, right, bottom


def get_body_part(rect: Rect):
    left = min(rect.A.x, rect.B.x)
    right = max(rect.A.x, rect.B.x)
    top = min(rect.A.y, rect.D.y)
    bottom = max(rect.A.y, rect.D.y)
    return left, top, right, bottom


def annotate_detections(image: np.ndarray, landmarks: t.List[glm.vec3], landmarksColor):
    if len(landmarks) == BlazePoseBase.total_count:
        for point in landmarks:
            image = cv2.circle(image, denormalized_point(image, spoint(point)), 2, landmarksColor, 1)

        image = annotate_rect(image, blaze_head(landmarks), (70, 150, 200))
        image = annotate_rect(image, blaze_left_wrist(landmarks), (200, 70, 150))
        image = annotate_rect(image, blaze_right_wrist(landmarks), (200, 70, 150))
    return image


def init_geometry(landmarks: t.List[glm.vec3]) -> Geometry:
    rect_head = blaze_head(landmarks)
    rect_hips = blaze_hips(landmarks)
    rect_body = blaze_torso(landmarks)
    rect_body_and_hips = blaze_torso_and_hips(landmarks)
    rect_right_foot = blaze_right_foot(landmarks)
    rect_left_foot = blaze_left_foot(landmarks)

    return Geometry(rect_head, rect_hips, rect_body, rect_body_and_hips, rect_right_foot, rect_left_foot)


def crop_image(source_image: np.ndarray, take_crops: bool = False, is_take_coords: bool = False) -> t.Union[dict, BodyCrops, tuple]:
    mp_pose = mp.solutions.pose
    landmarks: t.List[glm.vec3] = []

    with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
        image_height, image_width, _ = source_image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            result_dict = {}

            result_dict['nose'] = {'x': 0, 'y': 0, 'p': 0.9}
            result_dict['left_eye'] = {'x': 0, 'y': 0, 'p': 0.9}
            result_dict['right_eye'] = {'x': 0, 'y': 0, 'p': 0.9}
            result_dict['left_ear'] = {'x': 0, 'y': 0, 'p': 0.9}
            result_dict['right_ear'] = {'x': 0, 'y': 0, 'p': 0.9}
            result_dict['left_shoulder'] = {'x': 0, 'y': 0, 'p': 0.9}
            result_dict['right_shoulder'] = {'x': 0, 'y': 0, 'p': 0.9}
            result_dict['left_elbow'] = {'x': 0, 'y': 0, 'p': 0.9}
            result_dict['right_elbow'] = {'x': 0, 'y': 0, 'p': 0.9}
            result_dict['left_wrist'] = {'x': 0, 'y': 0, 'p': 0.9}
            result_dict['right_wrist'] = {'x': 0, 'y': 0, 'p': 0.9}
            result_dict['left_hip'] = {'x': 0, 'y': 0, 'p': 0.9}
            result_dict['right_hip'] = {'x': 0, 'y': 0, 'p': 0.9}
            result_dict['left_knee'] = {'x': 0, 'y': 0, 'p': 0.9}
            result_dict['right_knee'] = {'x': 0, 'y': 0, 'p': 0.9}
            result_dict['left_ankle'] = {'x': 0, 'y': 0, 'p': 0.9}
            result_dict['right_ankle'] = {'x': 0, 'y': 0, 'p': 0.9}
            result_dict['neck'] = {'x': 0, 'y': 0, 'p': 0.9}
            return result_dict
        print(
            f'Nose coordinates: ('
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
            f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
        )
        for keypoint in results.pose_landmarks.landmark:
            landmarks.append(glm.vec3(keypoint.x, keypoint.y, keypoint.z))

        if take_crops:
            geometry = init_geometry(landmarks)
            images = fill_images(source_image, 0, geometry)
            return images
        elif is_take_coords:
            geometry = init_geometry(landmarks)

            return get_person(geometry), get_body_part(geometry.rect_body), get_body_part(geometry.rect_body_and_hips),\
                   get_body_part(geometry.rect_hips), get_body_part(geometry.rect_left_foot), get_body_part(geometry.rect_right_foot),


        else:
            return blaze_all(landmarks)