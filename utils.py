"""
Save and load detection & tracking results

"""

def save_det_result(result_file, id_to_info, frame_no):
    for obj_id in id_to_info:
        class_id, score, bbox = id_to_info[obj_id]
        left, top, right, bottom = bbox
        result_file.write(f"{frame_no},{obj_id},{class_id},{score},{left},{top},{right},{bottom}\n")

def load_det_result(result_path):
    all_results = []
    last_frame_no = -1
    frame_results = []
    with open(result_path, "r") as result_file:
        for line in result_file:
            _line = line.strip()
            frame_no, obj_id, class_id, score, left, top, right, bottom = _line.split(",")
            frame_no, class_id = int(frame_no), int(class_id)
            score = float(score)
            left, top, right, bottom = int(left), int(top), int(right), int(bottom)
            
            if last_frame_no == -1:#first line
                last_frame_no = frame_no

            if frame_no>last_frame_no:
                # print(frame_no, frame_results)
                all_results.append(frame_results)
                for i in range(1, frame_no-last_frame_no):
                    all_results.append([])
                frame_results = []
                frame_results = []
                last_frame_no = frame_no

            frame_results.append([obj_id, class_id, score, left, top, right, bottom])

    all_results.append(frame_results) #last frame
    result_file.close()
    print(f"Loaded results of {len(all_results)} frames from {result_path}")
    return all_results


def use_det_result(all_results, frame_no):
    id_to_info = {}
    try:
        frame_results = all_results[frame_no-1]
        for (obj_id, class_id, score, left, top, right, bottom) in frame_results:
            info = [class_id, score, [left, top, right, bottom]]
            id_to_info[obj_id] = info
        return id_to_info
    except IndexError:
        return {}


"""
Computation

"""
def compute_overlapped(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    return interArea / boxAArea


def compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

# To check whether a point(x,y) is within a triangle area of interest
# by computer the 3 traingles form with any 2 point & (x,y)
# and check if the total area of the 3 traingles equal to the triangle of interest
def inside_roi(x,y, pts):
    x1, y1 = pts[0]
    x2, y2 = pts[1]
    x3, y3 = pts[2]
    a0 = area(x1, y1, x2, y2, x3, y3)
    a1 = area(x, y, x2, y2, x3, y3)
    a2 = area(x1, y1, x, y, x3, y3)
    a3 = area(x1, y1, x2, y2, x, y)
    if a1+a2+a3 == a0:
      return True
    else:
      return False


def area(x1, y1, x2, y2, x3, y3): 
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0) 


def euclidean_distance(x1,x2,y1,y2):
    return math.sqrt( (x2-x1)**2 + (y2-y1)**2)    