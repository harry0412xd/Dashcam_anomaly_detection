def detect_jaywalker(recent_bboxes, frame, mean_shift):
    if len(recent_bboxes)==0:
      return
    threshold = 0.5
    global vid_width, vid_height
    # Combined with lane detection could be better
    ROI = [(vid_width//4,vid_height), (vid_width//2,0), (vid_width*3//4, vid_height) ]
    cv2.polylines(frame, [np.array(ROI, dtype=np.int32)], False, (255,0,0))
    cv2.line(frame,(0, vid_height*2//5), (vid_width, vid_height*2//5), (255,0,0))

    score = 0
    max_score = len(recent_bboxes)
    x_diff_total = 0
    x_prev = -1
    is_moving_right = 0
    is_inside_ROI = False
    cur_center_x = 0
    cur_center_y = 0
    for bboxes_n_frameNum in recent_bboxes:
        left, top, right, bottom = bboxes_n_frameNum[0]
        offset = bboxes_n_frameNum[1]
        center_x, center_y = (left+right)//2, (top+bottom)//2
        if x_prev == -1:# current box i.e. first iteration
            is_inside_ROI = inside_roi(center_x, center_y, ROI)
            cur_center_x, cur_center_y = center_x, center_y
            if cur_center_x>vid_height//2:#Get the mean for the corr portion
                mean = mean_shift[1]
                is_left = True
            else:
                mean = mean_shift[0]
                is_left = False
        else:
            x_diff = center_x - x_prev
            if is_left: #The object is in the left portion
                if (x_diff>0): #more probably moving right
                    score += x_diff//vid_width 
                elif (x_diff-mean<0): #maybe moving left
                    move_percent = max((x_diff-mean)//vid_width - 0.05, 0) #ignore movement within 0.5% of frame width
                    score += x_diff//vid_width * 1.5

            else: #The object is in the right portion
                if (x_diff<0):#more probably moving left
                    score += abs(x_diff)//vid_width *100
                elif (x_diff-mean>0): #maybe moving right
                    move_percent = max(abs((x_diff-mean)//vid_width - 0.05, 0))
                    score += abs(x_diff)//vid_width * 1.5
            x_diff_total += x_diff        
        x_prev = center_x
    
    score += abs(x_diff_total)/len(recent_bboxes)*30
    
    if is_inside_ROI:
        if cur_center_x>(vid_height*2//5):
            score += max_score*0.2
        else:
            score *= 0.3
    else:
        score *= 0.01
    # print(f"score: {score} / {max_score} ")
    cv2.putText(frame, f"{(score/max_score):.2f} ", (cur_center_x-10, cur_center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    if score > threshold*max_score:
        return True
    return False
              

# get mean bbox shift
def get_mean_shift(frames_infos, out_frame):
    # seperate frame by left/right portion
    lp_shift_list, rp_shift_list = [], []
    lp_left_count, lp_right_count= 0, 0
    rp_left_count, rp_right_count= 0, 0

    id_list = [*frames_infos[0]] #what id to look for
    for obj_id in id_list:
        i = 1
        while i< len(frames_infos) and not obj_id in frames_infos[i]: #find the next frame containing obj 
            i += 1
        if i == len(frames_infos): #not found
            continue
        _, _, box_cur = frames_infos[0][obj_id]
        _, _, box_next = frames_infos[i][obj_id]
        left, top, right, bottom = box_cur
        left_next, top_next, right_next, bottom_next = box_next
        x_diff = (right_next+left_next)//2 - (right+left)//2 
        x_diff = x_diff/i  #diff per frame
        # print(f"Obj {obj_id}: {x_diff}")
        # cv2.putText(out_frame, f"{x_diff:.2f} ", ((right+left)//2, (top+bottom)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        if (right+left)>vid_width: #Object in right portion
            if x_diff>0 :
                rp_right_count += 1
            elif x_diff<0 :
                rp_left_count += 1
            if not x_diff==0:
                rp_shift_list.append(x_diff)
        else:
            if x_diff>0 :
                lp_right_count += 1
            elif x_diff<0 :
                lp_left_count += 1
            if not x_diff==0:
                lp_shift_list.append(x_diff)

    left_mean = cal_weighted_mean(lp_shift_list, lp_left_count, lp_right_count)
    cv2.putText(out_frame, f"{left_mean:.2f} ", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    right_mean = cal_weighted_mean(rp_shift_list, rp_left_count, rp_right_count)
    cv2.putText(out_frame, f"{right_mean:.2f} ", (vid_width-50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return left_mean, right_mean


def cal_weighted_mean(shift_list, left_count, right_count):
    if (left_count+right_count)==0:
        return 0
    shift_total = 0
    weight_left = left_count/(left_count+right_count)
    weight_right = right_count/(left_count+right_count)
    weight_total = 0
    for n in shift_list:
      if n>0:
          weight_total += weight_right
          shift_total += n * weight_right
      else:
          weight_total += weight_left  
          shift_total += n * weight_left

    return shift_total/weight_total