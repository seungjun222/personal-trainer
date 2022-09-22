import django
django.setup()
from scipy.spatial import distance as dist
import numpy as np
import pandas as pd
import progressbar
import cv2
import math
from .models import User
from . import models

# 운동 종류 , 유저(사용자인지,트레이너인지)를 입력받아 스켈레톤 동영상을 생성하고 좌표를 csv로 저장
def get_video(exercise_Type, path, id,request_id):
    
    #실시간 퍼센트 초기화
    t=models.Document.objects.get(id=request_id)
    t.time=0
    t.save()
    
    protoFile = "./model/pose_deploy_linevec.prototxt"
    weightsFile = "./model/pose_iter_160000.caffemodel"
    
    video_path ='.'+path

    out_path = './static/user_result/'+id+'.mp4'  # 결과 파일
    csv_path = './static/user_result/'+id+'.csv'  # 결과 파일 csv

    

    # 모델과 가중치 불러오기
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    # 비디오 정보 저장
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    ok, frame = cap.read()
    
    if exercise_Type=="스쿼트" or exercise_Type=="벤치프레스":
        frame = cv2.resize(frame, (640,360), cv2.INTER_AREA)#해상도 조절
    else:
        frame = cv2.resize(frame, (360,640), cv2.INTER_AREA)#풀업 해상도
        
    (frameHeight, frameWidth) = frame.shape[:2]
    print(frameHeight, frameWidth)
    h = frameHeight
    w = frameWidth

    # 모델에 입력을 크기
    inHeight = h
    inWidth = w

    # 아웃풋(스켈레톤 동영상 설정)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    output = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    writer = None
    (f_h, f_w) = (h, w)
    zeros = None

    data = []
    previous_x, previous_y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # 15개의 부위
    pairs = [[0, 1],  # 머리
             [1, 2], [1, 5],  # 어깨
             [2, 3], [3, 4], [5, 6], [6, 7],  # 팔
             [1, 14], [14, 11], [14, 8],  # 엉덩이
             [8, 9], [9, 10], [11, 12], [12, 13]]  # 다리
    

    bnp_pairs = [[2, 3], [3, 4],  # 벤치프레스 or 풀업 선택 시 부위
                 [5, 6], [6, 7]]

    bnp_points = [2, 3, 4, 5, 6, 7]


    sqt_pairs = [[0, 1],         # 스쿼트 선택 시 부위
                 [1, 14], [8, 14],
                 [8, 9], [9, 10]]

    sqt_points = [0, 1, 8, 9, 10, 14]



    # 임계값
    thresh = 0.1

    # 점,선 색
    circle_color, line_color = (0, 255, 255), (0, 255, 0)

    # 진행률 표시
    widgets = ["비디오 변환 진행률: ", progressbar.Percentage(), " ",
               progressbar.Bar(), " ", progressbar.ETA()]
    
    pbar = progressbar.ProgressBar(maxval=n_frames,
                                   widgets=widgets).start()
    p = 0
    # 관절 움직임 저장 리스트
    frame_xy = [[0], [1], [2], [3], [4], [5],
                [6], [7], [8], [9], [10], [11],
                [12], [13], [14]]

    # 시작
    #print(models.Document.objects.get(id=request_id).time)
    while True:
        ok, frame = cap.read()
        if ok != True:
            break

        frame = cv2.resize(frame, (w, h), cv2.INTER_AREA)
        frame_copy = np.copy(frame)

        # 네트워크 전처리
        inpBlob = cv2.dnn.blobFromImage(
            frame_copy, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()
        H = output.shape[2]
        W = output.shape[3]

        points = []
        x_data, y_data = [], []

        # 프레임별 데이터 저장 및 반복
        for i in range(15):
            probMap = output[0, i, :, :]
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
            x = (w * point[0]) / W
            y = (h * point[1]) / H

            if prob > thresh:
                points.append((int(x), int(y)))
                x_data.append(x)
                y_data.append(y)
            else:
                points.append((0, 0))
                x_data.append(previous_x[i])
                y_data.append(previous_y[i])

            #print(i," ",points[i][0], points[i][1],'',end='')


        if (exercise_Type == "벤치프레스" or exercise_Type == "풀업") :
            for pair in bnp_pairs:
                partA = pair[0]
                partB = pair[1]
                cv2.line(frame_copy, points[partA], points[partB],
                         line_color, 1, lineType=cv2.LINE_AA)

            for i in bnp_points:  # len(points)
                cv2.circle(frame_copy, (points[i][0],
                                        points[i][1]), 5, circle_color, -1)
                cv2.putText(frame_copy, str(
                    i), (points[i][0], points[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA)

                frame_xy[i].append((points[i][0], points[i][1]))

        elif exercise_Type == "스쿼트":
            for pair in sqt_pairs:
                partA = pair[0]
                partB = pair[1]
                cv2.line(frame_copy, points[partA], points[partB],
                         line_color, 1, lineType=cv2.LINE_AA)

            for i in sqt_points:  # len(points)
                cv2.circle(frame_copy, (points[i][0],
                                        points[i][1]), 5, circle_color, -1)
                cv2.putText(frame_copy, str(
                    i), (points[i][0], points[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, lineType=cv2.LINE_AA)

                frame_xy[i].append((points[i][0], points[i][1]))

        else:#아무 운동이 아닐시 모든 부위 딕텍션
            for pair in pairs:
                partA = pair[0]
                partB = pair[1]
                cv2.line(frame_copy, points[partA], points[partB],
                    line_color, 1, lineType=cv2.LINE_AA)
                

                
    
        if writer is None:
            writer = cv2.VideoWriter(out_path, fourcc, fps,
                                     (f_w, f_h), True)
            zeros = np.zeros((f_h, f_w), dtype="uint8")
        writer.write(cv2.resize(frame_copy, (f_w, f_h)))

        cv2.imshow('frame', frame_copy)

        data.append(x_data + y_data)
        previous_x, previous_y = x_data, y_data

        p += 1
        #print(models.Document.objects.get(id=request_id).time)
        pbar.update(p)
        
        t=models.Document.objects.get(id=request_id)
        t.time=int(pbar.percent)
        t.save()
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # csv로 변환
    df = pd.DataFrame(frame_xy)
    df = df.transpose()
    df = df.drop(0)
    df.to_csv(csv_path, index=False)
    
    pbar.finish()
    cap.release()
    cv2.destroyAllWindows()
    check(id)
    direct(exercise_Type, id)

#포인트 이동 중 (0,0)으로 튄 포인트 찾아서 보정 후 다시 저장, 필요없는 문자제거
def check(id):
    path = './static/user_result/'+id+'.csv'
    data = pd.read_csv(path)
    row,column =data.shape
    row=int(row)
    column=int(column)

    for i in range(column):
        for j in range(row):
            try:
                data[str(i)][j]=data[str(i)][j].replace("(","")#(제거
                data[str(i)][j]=data[str(i)][j].replace(")","")#)제거
                if data[str(i)][j] =='0, 0':#0,0이 있으면 이전 값 저장
                    data[str(i)][j]=data[str(i)][j-1]
            except:
                continue

                
                
    data.to_csv(path,header=True,index=False)#csv로 다시 저장

def angle(path,first,second,third):
    
    data = pd.read_csv(path)
    row,column=data.shape
    angle_list=[]
    
    #원하는 3지점 데이터 받아오기
    first_list=data[str(first)]
    second_list=data[str(second)]
    third_list=data[str(third)]
    
    for i in range(int(row)):
        #프레임별 위치 데이터 불러오기
        f_x,f_y=map(int,(first_list[i].split(",")))
        s_x,s_y=map(int,(second_list[i].split(",")))
        t_x,t_y=map(int,(third_list[i].split(",")))
    
        s_to_f=(f_x-s_x,f_y-s_y)#second-first
        s_to_t=(t_x-s_x,t_y-s_y)#second-third
        
        dot=s_to_f[0]*s_to_t[0]+s_to_f[1]*s_to_t[1]
        det=s_to_f[0]*s_to_t[1]-s_to_f[1]*s_to_t[0 ]
        
        theta=np.rad2deg(np.arctan2(det, dot))
        
        if theta < 0:
            theta=theta+360.0
            
        angle_list.append(theta)


    
    return angle_list

def quintile(data):
    quintile=[]
    quintile.append(np.min(data))
    quintile.append(np.percentile(data,25))
    quintile.append(np.percentile(data,50))
    quintile.append(np.percentile(data,75))
    quintile.append(np.max(data))
    
    return quintile
    
def benchpress_angle(path):
    
    left_arm = angle(path, 2,3,4) # 왼팔
    right_arm = angle(path, 7,6,5) # 오른팔
    
    #5분위수 저장-왼쪽 팔,오른쪽팔
    benchpress_left_angle=quintile(left_arm)
    benchpress_right_angle=quintile(right_arm)
    
    #5분위수 리턴
    return [benchpress_left_angle , benchpress_right_angle]

# 2. 스쿼트
def squat_angle(path):
    
    back_neck = angle(path, 14,1,0)#뒷목
    back_right_knee = angle(path, 10,9,8) # 오른쪽 뒷무릎
    spine = angle(path, 8,14,1)#척추
    
    #5분위수 계산
    squat_neck=quintile(back_neck)
    squat_right_knee=quintile(back_right_knee)
    squat_spine=quintile(spine)
    
    return [squat_neck,squat_right_knee,squat_spine]   
 
def range_motion(exercise_type,id):#유저,트레이너,운동종류
    user_path='./static/user_result/'+id+'.csv'
    
    if exercise_type=="스쿼트":
        trainer_path='./static/trainer/squat_trainer_result.csv'
        trainer_range=squat_angle(trainer_path)
        user_range=squat_angle(user_path)
        T_knee=max(trainer_range[1])-min(trainer_range[1])
        U_Knee=max(user_range[1])-min(user_range[1])
        result=round((U_Knee/T_knee)*100,1)
        
        return result
      
        
    elif exercise_type=="벤치프레스":
        trainer_path='./static/trainer/benchpress_trainer_result.csv'
        trainer_range=benchpress_angle(trainer_path)
        user_range=benchpress_angle(user_path)
        T_right_angle=max(trainer_range[1])-min(trainer_range[1])
        T_left_angle=max(trainer_range[0])-min(trainer_range[0])
        U_right_angle=max(user_range[1])-min(user_range[1])
        U_left_angle=max(user_range[0])-min(user_range[0])
        
        left_result=round((U_left_angle/T_left_angle)*100,1)
        right_result=round((U_right_angle/T_right_angle)*100,1)

        return left_result,right_result
    
    else:
        trainer_path='./static/trainer/pullup_trainer_result.csv'
        trainer_range=benchpress_angle(trainer_path)
        user_range=benchpress_angle(user_path)
        T_right_angle=max(trainer_range[1])-min(trainer_range[1])
        T_left_angle=max(trainer_range[0])-min(trainer_range[0])
        U_right_angle=max(user_range[1])-min(user_range[1])
        U_left_angle=max(user_range[0])-min(user_range[0])
        
        left_result=round((U_left_angle/T_left_angle)*100,1)
        right_result=round((U_right_angle/T_right_angle)*100,1)

        return left_result,right_result
    
def direct(exercise_Type, id):
    path='./static/user_result/'+id+'.csv'
    data = pd.read_csv(path) #check path
    row_count=len(data) #프레임수
    df_new=pd.DataFrame()
    
    for i in range(0,15):
        df=pd.DataFrame(columns=["id",i])
        for j in range(0, row_count-1):
            
            #처음 좌표
            data_1=data.iat[j,i]
            if pd.isnull(data_1):
                continue
            
            x1,y1=data_1.split(", ")
            x1,y1=int(x1),int(y1)
        
            data_2=data.iat[j+1,i]
            x2,y2=data_2.split(", ")
            x2,y2=int(x2),int(y2)
            
            a=math.atan2(-(y2-y1),x2-x1)*(180/math.pi) # 방위각 계산, cv좌표계 -> y축변환
            
            if x1==x2 and y1==y2:
                direct="" #움직임 변화가 없을때
            elif -22.5<=a<=22.5: 
                direct=1 # →
            elif 22.5<a<=67.5: 
                direct=2 # ↗
            elif 67.5<a<=112.5:
                direct=3 # ↑
            elif 112.5<a<=157.5:
                direct=4 # ↖
            elif -180<a<=-157.5 or 157.5<a<=180:
                direct=5 # ←
            elif -157.5<a<=-112.5: 
                direct=6 # ↙
            elif -112.5<a<=-67.5: 
                direct=7 # ↓
            elif -67.5<a<-22.5:
                direct=8 #↘
        
            df= df.append(pd.DataFrame([[j,direct]], columns=["id",i]), ignore_index=True)
        df_new=df_new.append(df[i])
    
    df_new=df_new.transpose()
    
    if exercise_Type=='스쿼트':
        df_new.to_csv('./static/user_result/'+id+'_squat_direct.csv', index = False)
    elif exercise_Type=='벤치프레스':
        df_new.to_csv('./static/user_result/'+id+'_benchpress_direct.csv', index = False)
    else:
        df_new.to_csv('./static/user_result/'+id+'_pullup_direct.csv', index = False)

def move_direct(exercise_Type, id, path, measure_body):
    BODY_PARTS = {0:"머리", 1:"목", 2:"오른쪽 어깨", 3:"오른쪽 팔꿈치", 4:"오른쪽 팔목",
              5:"왼쪽 어깨", 6:"왼쪽 팔꿈치", 7:"왼쪽 팔목", 8:"오른쪽 엉덩이", 9:"오른쪽 무릎",
              10:"오른쪽 발목", 11:"왼쪽 엉덩이", 12:"왼쪽 무릎", 13:"왼쪽 발목", 14:"허리",
              15:"Background"}

    data = pd.read_csv(path)
    b=[]
    barchart_part=[]
    barchart_data=[]
    for i in measure_body:
        list=[]
        
        for j in range(0,len(data)-1):
            direct=data.iat[j,i]
            
            if direct==1: direct="→"
            elif direct==2: direct="↗"
            elif direct==3: direct="↑"
            elif direct==4: direct="↖"
            elif direct==5: direct="←"
            elif direct==6: direct="↙"
            elif direct==7: direct="↓"
            elif direct==8: direct="↘"
            
            if pd.notnull(direct):
                list.append(direct)
            for k in range(0,len(list)-1):
                if list[k]==list[k+1]:
                    del list[k+1]
        a=str("[{}] direct: {} 방향 변화점 수: {}").format(BODY_PARTS[i],list,len(list)-1)
        barchart_part.append(BODY_PARTS[i])
        barchart_data.append(len(list)-1)
        #print(a)
        b.append(a)
    result=""    
    
    for i in b:
        result=result+i+"\n"
        
    return result,barchart_part,barchart_data
    
def direct_detail(exercise_Type, id):
    if exercise_Type=="벤치프레스":
        measure_body = [5, 6, 7, 2, 3, 4] #4,7:팔목 3,6:팔꿈치 2,5:어깨
        trainer_path='./static/trainer/benchpress_trainer_result_direct.csv'
        user_path='./static/user_result/'+id+'_benchpress_direct.csv'
    elif exercise_Type=="스쿼트":
        measure_body = [8, 9, 14] #8:엉덩이 9:무릎 14:허리
        trainer_path='./static/trainer/squat_trainer_result_direct.csv'
        user_path='./static/user_result/'+id+'_squat_direct.csv'
    else:
        measure_body = [5, 6, 2, 3] #4,7:팔목 3,6:팔꿈치 2,5:어깨
        trainer_path='./static/trainer/pullup_trainer_result_direct.csv'
        user_path='./static/user_result/'+id+'_pullup_direct.csv'
        
    trainer_result,trainer_barchart_part,trainer_barchart_data=move_direct(exercise_Type,id,trainer_path,measure_body)
    user_result,user_barchart_part,user_barchart_data=move_direct(exercise_Type,id,user_path,measure_body) 

    return trainer_result,user_result,trainer_barchart_part,trainer_barchart_data,user_barchart_part,user_barchart_data


def feedback_str(exercise_Type,box,part,first,second,third,last):#4개 구간 점수와 부위를 입력받아 부족한 부분 피드백
    score=[first,second,third,last]
    check_box=[0,0,0,0]#왼팔 오른팔 부분 비교
    
    if exercise_Type=="스쿼트":
        if part=="뒷목":
            if any( (num<90 or num>110) for num in score):#score값 중 하나라도 90미만 이거나 110초과면
                box.append("목이 너무 꺾이지 않았는지 확인해보세요!")
            
        elif part=="무릎":
            if first<90 or first>110 or second<90 or second>110:#90~110 사이에 없을 때
                if first> 100 or second>  100: #100보다 큰가 작은가. 크면 트레이너대비 많이 움직인 수치
                    box.append("트레이너대비 "+str(part)+"이 내려갈때 각도가 너무 벌어져요.")   
                else:
                    box.append("트레이너대비 "+str(part)+"이 내려갈때 각도가 너무 좁아져요.") 
                        
            if third<90 or third>110 or last<90 or last>110:#90~110 사이에 없을 때
                if third> 100 or last>  100: #100보다 큰가 작은가. 크면 트레이너대비 많이 움직인 수치
                    box.append("트레이너대비 "+str(part)+"이 올라갈때 각도가 너무 벌어져요.")   
                else:
                    box.append("트레이너대비 "+str(part)+"이 올라갈때 각도가 너무 좁아져요.")   
            
        elif part=="척추":
            if any( (num<90 or num>110) for num in score):#score값 중 하나라도 90미만 이거나 110초과면
                box.append("허리가 굽지 않았는지 확인해보세요!")
                
    elif exercise_Type=="벤치프레스":
        if first<90 or first>110 or second<90 or second>110:#90~110 사이에 없을 때
            if first<90 or first>110:
                check_box[0]=1
            if second<90 or second>110:
                check_box[1]=1
                
            if first> 100 or second>  100: #100보다 큰가 작은가. 크면 트레이너대비 많이 움직인 수치
                box.append("트레이너대비 "+str(part)+"이 내려갈때 각도가 너무 벌어져요.")   
            else:
                box.append("트레이너대비 "+str(part)+"이 내려갈때 각도가 너무 좁아져요.") 
                    
        if third<90 or third>110 or last<90 or last>110:#90~110 사이에 없을 때
            if third<90 or third>110:
                check_box[2]=1
            if last<90 or last>110:
                check_box[3]=1
            if third> 100 or last>  100: #100보다 큰가 작은가. 크면 트레이너대비 많이 움직인 수치
                box.append("트레이너대비 "+str(part)+"이 올라갈때 각도가 너무 벌어져요.")   
            else:
                box.append("트레이너대비 "+str(part)+"이 올라갈때 각도가 너무 좁아져요.") 
    else:#풀업
        if first<90 or first>110 or second<90 or second>110:#90~110 사이에 없을 때
            if first<90 or first>110:
                check_box[0]=1
            if second<90 or second>110:
                check_box[1]=1
                
            if first> 100 or second>100: #100보다 큰가 작은가. 크면 트레이너대비 많이 움직인 수치
                box.append("트레이너대비 "+str(part)+"이 올라갈때 각도가 너무 벌어져요.")   
            else:
                box.append("트레이너대비 "+str(part)+"이 올라갈때 각도가 너무 좁아져요.") 
                    
        if third<90 or third>110 or last<90 or last>110:#90~110 사이에 없을 때
            if third<90 or third>110:
                check_box[2]=1
            if last<90 or last>110:
                check_box[3]=1
            if third> 100 or last>  100: #100보다 큰가 작은가. 크면 트레이너대비 많이 움직인 수치
                box.append("트레이너대비 "+str(part)+"이 내려갈때 각도가 너무 벌어져요.")   
            else:
                box.append("트레이너대비 "+str(part)+"이 내려갈때 각도가 너무 좁아져요.")
    return check_box
        

def angle_change(exercise_Type,id):#유저,트레이너,운동종류
    
    grade=[]#점수
    detail=[]#자세히보기 기능
    feedback=[]
    
    if exercise_Type=="스쿼트":
        user_path='./static/user_result/'+id+'.csv'
        trainer_path='./static/trainer/squat_trainer_result.csv'
        detail.append("<스쿼트>-각도변화량")
        T_back_neck = angle(trainer_path, 14,1,0)#뒷목
        U_back_neck = angle(user_path,14,1,0)
        detail.append("뒷목 각도 변화")
        first,second,third,last=angle_parts(T_back_neck,U_back_neck)
        feedback_str(exercise_Type,feedback,"뒷목",first,second,third,last)#피드백 한글
        listplus(detail,first,second,third,last)
        grade.append(score(first));grade.append(score(second));grade.append(score(third));grade.append(score(last))
   
        T_back_right_knee = angle(trainer_path, 10,9,8) # 오른쪽 뒷무릎
        U_back_right_knee = angle(user_path, 10,9,8)
        detail.append("무릎 각도 변화")
        first,second,third,last=angle_parts(T_back_right_knee,U_back_right_knee)
        feedback_str(exercise_Type,feedback,"무릎",first,second,third,last)#피드백 한글
        listplus(detail,first,second,third,last)
        grade.append(score(first));grade.append(score(second));grade.append(score(third));grade.append(score(last))

        
        T_spine = angle(trainer_path, 8,14,1)#척추
        U_spine= angle(user_path, 8,14,1)#척추
        detail.append("척추 각도 변화:")
        first,second,third,last=angle_parts(T_spine,U_spine)
        feedback_str(exercise_Type,feedback,"척추",first,second,third,last)#피드백 한글
        listplus(detail,first,second,third,last)
        grade.append(score(first));grade.append(score(second));grade.append(score(third));grade.append(score(last))
        k=""
        
        for i in detail:
            k=k+i+"\n"
        
    elif exercise_Type=="벤치프레스":
        user_path='./static/user_result/'+id+'.csv'
        trainer_path='./static/trainer/benchpress_trainer_result.csv'
        detail.append("<벤치프레스>-각도변화량")
        T_left_arm = angle(trainer_path, 2,3,4) # 왼팔
        T_right_arm = angle(trainer_path, 7,6,5) # 오른팔
        U_left_arm = angle(user_path, 2,3,4) # 왼팔
        U_right_arm = angle(user_path, 7,6,5) # 오른팔
        detail.append("왼팔 각도 변화")
        first,second,third,last= angle_parts(T_left_arm,U_left_arm)
        check_left=feedback_str(exercise_Type,feedback,"왼팔",first,second,third,last)#피드백 한글
        listplus(detail,first,second,third,last)
        grade.append(score(first));grade.append(score(second));grade.append(score(third));grade.append(score(last))
        detail.append("오른팔 각도 변화")
        first,second,third,last= angle_parts(T_right_arm,U_right_arm)
        check_right=feedback_str(exercise_Type,feedback,"오른팔",first,second,third,last)#피드백 한글
        listplus(detail,first,second,third,last)
        grade.append(score(first));grade.append(score(second));grade.append(score(third));grade.append(score(last))
        k=""
        for i in detail:
            k=k+i+"\n"
        if check_right != check_left:
            feedback.append("트레이너대비 "+ "양팔이 불균형 합니다. 양팔 대칭으로 운동해 보세요.")

    else:#풀업
        user_path='./static/user_result/'+id+'.csv'
        trainer_path='./static/trainer/benchpress_trainer_result.csv'
        detail.append("<풀업>-각도변화량")
        T_left_arm = angle(trainer_path, 7,6,5) # 왼팔
        T_right_arm = angle(trainer_path, 2,3,4) # 오른팔
        U_left_arm = angle(user_path, 7,6,5) # 왼팔
        U_right_arm = angle(user_path, 2,3,4) # 오른팔
        detail.append("왼팔 각도 변화")
        first,second,third,last= angle_parts(T_left_arm,U_left_arm)
        check_left=feedback_str(exercise_Type,feedback,"왼팔",first,second,third,last)#피드백 한글
        listplus(detail,first,second,third,last)
        grade.append(score(first));grade.append(score(second));grade.append(score(third));grade.append(score(last))
        detail.append("오른팔 각도 변화")
        first,second,third,last= angle_parts(T_right_arm,U_right_arm)
        check_right=feedback_str(exercise_Type,feedback,"오른팔",first,second,third,last)#피드백 한글
        listplus(detail,first,second,third,last)
        grade.append(score(first));grade.append(score(second));grade.append(score(third));grade.append(score(last))
        
        k=""
        for i in detail:
            k=k+i+"\n"
        if check_right != check_left:
            feedback.append("트레이너대비 "+"양팔이 불균형 합니다. 양팔 대칭으로 운동해 보세요.")   
   
    feedback_string=""
    for i in feedback:
        feedback_string=feedback_string+str(i)+"\n"
    return round(np.mean(grade),1), k ,feedback_string

#4구간 평균 각도 비교
def angle_parts(data,data2):#트레이너데이터, 유저 데이터
    T_data_length=len(data)
    
    T_first=np.mean(data[0:int(T_data_length*0.25)])
    T_second= np.mean(data[int(T_data_length*0.25):int(T_data_length*0.5)])
    T_third=np.mean(data[int(T_data_length*0.5):int(T_data_length*0.75)])
    T_last=np.mean(data[int(T_data_length*0.75):T_data_length-1])
    
    U_data_length=len(data2)
    U_first=np.mean(data2[0:int(U_data_length*0.25)])
    U_second= np.mean(data2[int(U_data_length*0.25):int(U_data_length*0.5)])
    U_third=np.mean(data2[int(U_data_length*0.5):int(U_data_length*0.75)])
    U_last=np.mean(data2[int(U_data_length*0.75):U_data_length-1])
    
    first_result=round(U_first/T_first*100,1)
    second_result=round(U_second/T_second*100,1)
    third_result=round(U_third/T_third*100,1)
    last_result=round(U_last/T_last*100,1)
    
    return first_result,second_result,third_result,last_result
    
    
#승준===============================================================================================================
def SJ_angle_parts(data, data2):#트레이너데이터, 유저 데이터
    T_data_length=len(data)
    
    T_first=np.mean(data[0:int(T_data_length*0.25)])
    T_second= np.mean(data[int(T_data_length*0.25):int(T_data_length*0.5)])
    T_third=np.mean(data[int(T_data_length*0.5):int(T_data_length*0.75)])
    T_last=np.mean(data[int(T_data_length*0.75):T_data_length-1])
    
    U_data_length=len(data2)
    U_first=np.mean(data2[0:int(U_data_length*0.25)])
    U_second= np.mean(data2[int(U_data_length*0.25):int(U_data_length*0.5)])
    U_third=np.mean(data2[int(U_data_length*0.5):int(U_data_length*0.75)])
    U_last=np.mean(data2[int(U_data_length*0.75):U_data_length-1])
    
    return T_first, T_second, T_third, T_last, U_first, U_second, U_third, U_last

def SJ_angle_change(exercise_Type,id):#유저,트레이너,운동종류
    
    if exercise_Type=="스쿼트":
        user_path='./static/user_result/'+id+'.csv'
        trainer_path='./static/trainer/squat_trainer_result.csv'
  
        T_back_neck = angle(trainer_path, 14,1,0)#뒷목
        U_back_neck = angle(user_path,14,1,0)
        T_back_neck_first,T_back_neck_second,T_back_neck_third,T_back_neck_last,U_back_neck_first,U_back_neck_second,U_back_neck_third,U_back_neck_last=SJ_angle_parts(T_back_neck,U_back_neck)
       
        T_back_right_knee = angle(trainer_path, 10,9,8) # 오른쪽 뒷무릎
        U_back_right_knee = angle(user_path, 10,9,8)
        T_back_right_knee_first,T_back_right_knee_second,T_back_right_knee_third,T_back_right_knee_last,U_back_right_knee_first,U_back_right_knee_second,U_back_right_knee_third,U_back_right_knee_last=SJ_angle_parts(T_back_right_knee,U_back_right_knee)
       
        T_spine = angle(trainer_path, 8,14,1)#척추
        U_spine= angle(user_path, 8,14,1)#척추
        T_spine_first,T_spine_second,T_spine_third,T_spine_last,U_spine_first,U_spine_second,U_spine_third,U_spine_last=SJ_angle_parts(T_spine,U_spine)
       
        return T_back_neck_first,T_back_neck_second,T_back_neck_third,T_back_neck_last,U_back_neck_first,U_back_neck_second,U_back_neck_third,U_back_neck_last,T_back_right_knee_first,T_back_right_knee_second,T_back_right_knee_third,T_back_right_knee_last,U_back_right_knee_first,U_back_right_knee_second,U_back_right_knee_third,U_back_right_knee_last,T_spine_first,T_spine_second,T_spine_third,T_spine_last,U_spine_first,U_spine_second,U_spine_third,U_spine_last
        
    elif exercise_Type=="벤치프레스": #
        user_path='./static/user_result/'+id+'.csv'
        trainer_path='./static/trainer/benchpress_trainer_result.csv'

        T_left_arm = angle(trainer_path, 2,3,4) # 왼팔
        T_right_arm = angle(trainer_path, 7,6,5) # 오른팔
        U_left_arm = angle(user_path, 2,3,4) # 왼팔
        U_right_arm = angle(user_path, 7,6,5) # 오른팔
        
        T_left_arm_first, T_left_arm_second, T_left_arm_third, T_left_arm_last, U_left_arm_first, U_left_arm_second, U_left_arm_third, U_left_arm_last= SJ_angle_parts(T_left_arm,U_left_arm)
        T_right_arm_first, T_right_arm_second, T_right_arm_third, T_right_arm_last, U_right_arm_first, U_right_arm_second, U_right_arm_third, U_right_arm_last= SJ_angle_parts(T_right_arm,U_right_arm)
    
        return T_left_arm_first, T_left_arm_second, T_left_arm_third, T_left_arm_last, U_left_arm_first, U_left_arm_second, U_left_arm_third, U_left_arm_last, T_right_arm_first, T_right_arm_second, T_right_arm_third, T_right_arm_last, U_right_arm_first, U_right_arm_second, U_right_arm_third, U_right_arm_last
    
    else:#풀업
        user_path='./static/user_result/'+id+'.csv'
        trainer_path='./static/trainer/benchpress_trainer_result.csv'
       
        T_left_arm = angle(trainer_path, 7,6,5) # 왼팔
        T_right_arm = angle(trainer_path, 2,3,4) # 오른팔
        U_left_arm = angle(user_path, 7,6,5) # 왼팔
        U_right_arm = angle(user_path, 2,3,4) # 오른팔
       
        T_left_arm_first, T_left_arm_second, T_left_arm_third, T_left_arm_last, U_left_arm_first, U_left_arm_second, U_left_arm_third, U_left_arm_last= SJ_angle_parts(T_left_arm,U_left_arm)
        T_right_arm_first, T_right_arm_second, T_right_arm_third, T_right_arm_last, U_right_arm_first, U_right_arm_second, U_right_arm_third, U_right_arm_last= SJ_angle_parts(T_right_arm,U_right_arm)
    
        return T_left_arm_first, T_left_arm_second, T_left_arm_third, T_left_arm_last, U_left_arm_first, U_left_arm_second, U_left_arm_third, U_left_arm_last, T_right_arm_first, T_right_arm_second, T_right_arm_third, T_right_arm_last, U_right_arm_first, U_right_arm_second, U_right_arm_third, U_right_arm_last
   
#==================================================================================================================


#4개구간 저장 함수    
def listplus(box,data1,data2,data3,data4):
    box.append("트레이너 대비 0%~25% 구간 평균각도 비율: "+ str(data1))
    box.append("트레이너 대비 25%~50% 구간 평균각도 비율: "+ str(data2))
    box.append("트레이너 대비 50%~75% 구간 평균각도 비율: "+ str(data3))
    box.append("트레이너 대비 75%~100% 구간 평균각도 비율: "+ str(data4))
    
#점수화
def score(data):
    if data<=100:
        return data
    else:
        return 200-data      
