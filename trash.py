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
              

