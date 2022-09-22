from os import P_DETACH
from django.http import request
from django.shortcuts import redirect, render
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import redirect
from .models import User
from . import models
from users import backend
from django.contrib.sessions.models import Session
from django.db.models import Q
from multiprocessing import Process
import json

def login_view(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]
        user = authenticate(username=username, password=password)

        if user is not None:
            print("인증 성공")
            login(request, user)
        else:
            print("인증 실패")

    return render(request, "users/login.html")


def logout_view(request):
    logout(request)
    return redirect("user:login")


def buttons_view(request):

    return render(request, "users/buttons.html")


def button1_view(request):

    return render(request, "users/button1.html")


def button2_view(request):

    return render(request, "users/button2.html")

def signup_view(request):
    if request.method == "POST":
        print(request.POST)
        # profile_img = request.FILES["profile_img"]
        username = request.POST["username"]
        password = request.POST["password"]
        realname = request.POST["realname"]
        email = request.POST["email"]
        # student_id = request.POST["student_id"]

        user = User.objects.create_user(username, email, password)
        user.realname = realname
        # user.student_id = student_id
        # user.profile_img = profile_img
        user.save()

        return redirect("user:login")

    return render(request, "users/signup.html")

  
def uploadFile(request):

    if request.method == "POST":
        # Fetching the form data
        request_id = request.user
        #fileTitle = request.POST["fileTitle"]
        uploadedFile = request.FILES["uploadedFile"]
        
        #사용자가 선택한 운동 종류에 따라 트레이너 영상 경로 지정
        if str(request.POST['workout'])== "벤치프레스":
            trainer_path='/static/trainer/benchpress_trainer.mp4'
        elif str(request.POST['workout'])== "스쿼트":
            trainer_path='/static/trainer/squat_trainer.mp4'
        elif str(request.POST['workout'])== "풀업":
            trainer_path='/static/trainer/pullup_trainer.mp4'
            
        request.session['exercise_type'] = str(request.POST['workout'])#사용자가 선택한 운동 종류 저장

        # db에 정보 저장
        document = models.Document(
            id=request_id,
            trainerpath=trainer_path,
            uploadedFile=uploadedFile
        )

        document.save()
        
        #스켈레톤 동영상과 csv 얻기
        
        #오랜 시간이 걸려 멀티 프로세싱으로 진행-백그라운드 작업
        p=Process(target=backend.get_video, args=(str(request.POST['workout']),
                                str(models.Document.objects.get(id=request_id).uploadedFile.url),#사용자 경로
                                str(request.user),request_id))
        p.start()
        # backend.get_video(str(request.POST['workout']),
        #                         str(models.Document.objects.get(id=request_id).uploadedFile.url),#사용자 경로
        #                         str(request.user)) #(280,80)
        #backend.check(str(request.user))#전처리
        #backend.direct(request.session['exercise_type'],str(request.user))
        return redirect("time/")
    
    return render(request, "users/upload-file.html")

def file_complete(request):
    context = {
            "path": str(models.Document.objects.get(id=request.user).uploadedFile.url),#사용자 경로
            'trainer_path':models.Document.objects.get(id=request.user).trainerpath,#유저경로
        }
    return render(request, "users/file-complete.html",context)

def feedback_view(request):
    
    exercise = str(request.session.get('exercise_type', False))#사용자가 선택한 운동종류 가져오기
    user_video = "/static/user_result/"+str(request.user)+'.mp4'
    trainer_video=''
    
    #트레이너 동영상 경로지정
    if str(request.session.get('exercise_type', False))=="스쿼트":
        trainer_video='/static/trainer/squat_trainer_result.mp4'
        op_range=backend.range_motion(exercise,str(request.user))
        if 80<=float(op_range)<=120:
            op_range= backend.range_motion(exercise,str(request.user))            
            score,angle_variation,feedback=backend.angle_change(exercise,str(request.user))
            trainer_result,user_result,trainer_barchart_part,trainer_barchart_data,user_barchart_part,user_barchart_data=backend.direct_detail(exercise,str(request.user))
            direct_variation="\n<스쿼트>-방향변화, 방향변화점 수\n트레이너\n"+trainer_result+"\n유저\n"+user_result
            
            #승준==============================================================
            
            SJ_a_v_0=backend.SJ_angle_change(exercise,str(request.user))[0]
            SJ_a_v_1=backend.SJ_angle_change(exercise,str(request.user))[1]
            SJ_a_v_2=backend.SJ_angle_change(exercise,str(request.user))[2]
            SJ_a_v_3=backend.SJ_angle_change(exercise,str(request.user))[3]
            SJ_a_v_4=backend.SJ_angle_change(exercise,str(request.user))[4]
            SJ_a_v_5=backend.SJ_angle_change(exercise,str(request.user))[5]
            SJ_a_v_6=backend.SJ_angle_change(exercise,str(request.user))[6]
            SJ_a_v_7=backend.SJ_angle_change(exercise,str(request.user))[7]
            SJ_a_v_8=backend.SJ_angle_change(exercise,str(request.user))[8]
            SJ_a_v_9=backend.SJ_angle_change(exercise,str(request.user))[9]
            SJ_a_v_10=backend.SJ_angle_change(exercise,str(request.user))[10]
            SJ_a_v_11=backend.SJ_angle_change(exercise,str(request.user))[11]
            SJ_a_v_12=backend.SJ_angle_change(exercise,str(request.user))[12]
            SJ_a_v_13=backend.SJ_angle_change(exercise,str(request.user))[13]
            SJ_a_v_14=backend.SJ_angle_change(exercise,str(request.user))[14]
            SJ_a_v_15=backend.SJ_angle_change(exercise,str(request.user))[15]
            
            SJ_a_v_16=backend.SJ_angle_change(exercise,str(request.user))[16]
            SJ_a_v_17=backend.SJ_angle_change(exercise,str(request.user))[17]
            SJ_a_v_18=backend.SJ_angle_change(exercise,str(request.user))[18]
            SJ_a_v_19=backend.SJ_angle_change(exercise,str(request.user))[19]
            SJ_a_v_20=backend.SJ_angle_change(exercise,str(request.user))[20]
            SJ_a_v_21=backend.SJ_angle_change(exercise,str(request.user))[21]
            SJ_a_v_22=backend.SJ_angle_change(exercise,str(request.user))[22]
            SJ_a_v_23=backend.SJ_angle_change(exercise,str(request.user))[23]
           
            context = {
                'exercise': exercise,
                'op_range': op_range,
                'score': score,
                'user_video': user_video,
                'trainer_video':trainer_video,
                'angle_variation': angle_variation,
                'direct_variation': direct_variation,
                'feedback' : feedback,
                
                'SJ_a_v_0':SJ_a_v_0,
                'SJ_a_v_1':SJ_a_v_1,
                'SJ_a_v_2':SJ_a_v_2,
                'SJ_a_v_3':SJ_a_v_3,
                'SJ_a_v_4':SJ_a_v_4,
                'SJ_a_v_5':SJ_a_v_5,
                'SJ_a_v_6':SJ_a_v_6,
                'SJ_a_v_7':SJ_a_v_7,
                'SJ_a_v_8':SJ_a_v_8,
                'SJ_a_v_9':SJ_a_v_9,
                'SJ_a_v_10':SJ_a_v_10,
                'SJ_a_v_11':SJ_a_v_11,
                'SJ_a_v_12':SJ_a_v_12,
                'SJ_a_v_13':SJ_a_v_13,
                'SJ_a_v_14':SJ_a_v_14,
                'SJ_a_v_15':SJ_a_v_15,
                
                'SJ_a_v_16':SJ_a_v_16,
                'SJ_a_v_17':SJ_a_v_17,
                'SJ_a_v_18':SJ_a_v_18,
                'SJ_a_v_19':SJ_a_v_19,
                'SJ_a_v_20':SJ_a_v_20,
                'SJ_a_v_21':SJ_a_v_21,
                'SJ_a_v_22':SJ_a_v_22,
                'SJ_a_v_23':SJ_a_v_23,
                'trainer_barchart_data':trainer_barchart_data,
                'user_barchart_data':user_barchart_data
                 }
           
        else:
            op_range="(가동범위 부족)"
            score="(측정 불가)"
            angle_variation = ""
            direct_variation = ""
            feedback="가동범위가 부족합니다."
            trainer_result=""
            user_result=""
            trainer_barchart_data=""
            user_barchart_data=""
            
            context = {
                'exercise': exercise,
                'op_range': op_range,
                'score': score,
                'user_video': user_video,
                'trainer_video':trainer_video,
                'angle_variation': angle_variation,
                'direct_variation': direct_variation,
                'feedback' : feedback,
                
                'SJ_a_v_0':0,
                'SJ_a_v_1':0,
                'SJ_a_v_2':0,
                'SJ_a_v_3':0,
                'SJ_a_v_4':0,
                'SJ_a_v_5':0,
                'SJ_a_v_6':0,
                'SJ_a_v_7':0,
                'SJ_a_v_8':0,
                'SJ_a_v_9':0,
                'SJ_a_v_10':0,
                'SJ_a_v_11':0,
                'SJ_a_v_12':0,
                'SJ_a_v_13':0,
                'SJ_a_v_14':0,
                'SJ_a_v_15':0,
                
                'SJ_a_v_16':0,
                'SJ_a_v_17':0,
                'SJ_a_v_18':0,
                'SJ_a_v_19':0,
                'SJ_a_v_20':0,
                'SJ_a_v_21':0,
                'SJ_a_v_22':0,
                'SJ_a_v_23':0,
                'trainer_barchart_data':trainer_barchart_data,
                'user_barchart_data':user_barchart_data
                }
            

    elif str(request.session.get('exercise_type', False))=="벤치프레스":
        trainer_video='/static/trainer/benchpress_trainer_result.mp4'
        bench_result_l,  bench_result_r=backend.range_motion(exercise,str(request.user))
        op_range="왼쪽: "+str( bench_result_l)+", 오른쪽: "+str( bench_result_r)
        if 80<= bench_result_l<120 and 80<= bench_result_r<120:
            op_range= backend.range_motion(exercise,str(request.user))
            score,angle_variation,feedback=backend.angle_change(exercise,str(request.user))
            trainer_result,user_result,trainer_barchart_part,trainer_barchart_data,user_barchart_part,user_barchart_data=backend.direct_detail(exercise,str(request.user))
            direct_variation="\n<벤치프레스>-방향 변화, 방향 변화점 수\n트레이너\n"+trainer_result+"\n유저\n"+user_result
            
            #승준==============================================================
            
            SJ_a_v_0=backend.SJ_angle_change(exercise,str(request.user))[0]
            SJ_a_v_1=backend.SJ_angle_change(exercise,str(request.user))[1]
            SJ_a_v_2=backend.SJ_angle_change(exercise,str(request.user))[2]
            SJ_a_v_3=backend.SJ_angle_change(exercise,str(request.user))[3]
            SJ_a_v_4=backend.SJ_angle_change(exercise,str(request.user))[4]
            SJ_a_v_5=backend.SJ_angle_change(exercise,str(request.user))[5]
            SJ_a_v_6=backend.SJ_angle_change(exercise,str(request.user))[6]
            SJ_a_v_7=backend.SJ_angle_change(exercise,str(request.user))[7]
            SJ_a_v_8=backend.SJ_angle_change(exercise,str(request.user))[8]
            SJ_a_v_9=backend.SJ_angle_change(exercise,str(request.user))[9]
            SJ_a_v_10=backend.SJ_angle_change(exercise,str(request.user))[10]
            SJ_a_v_11=backend.SJ_angle_change(exercise,str(request.user))[11]
            SJ_a_v_12=backend.SJ_angle_change(exercise,str(request.user))[12]
            SJ_a_v_13=backend.SJ_angle_change(exercise,str(request.user))[13]
            SJ_a_v_14=backend.SJ_angle_change(exercise,str(request.user))[14]
            SJ_a_v_15=backend.SJ_angle_change(exercise,str(request.user))[15]
            
            SJ_a_v_16=0
            SJ_a_v_17=0
            SJ_a_v_18=0
            SJ_a_v_19=0
            SJ_a_v_20=0
            SJ_a_v_21=0
            SJ_a_v_22=0
            SJ_a_v_23=0
            
            context = {
                'exercise': exercise,
                'op_range': op_range,
                'score': score,
                'user_video': user_video,
                'trainer_video':trainer_video,
                'angle_variation': angle_variation,
                'direct_variation': direct_variation,
                'feedback' : feedback,
                
                'SJ_a_v_0':SJ_a_v_0,
                'SJ_a_v_1':SJ_a_v_1,
                'SJ_a_v_2':SJ_a_v_2,
                'SJ_a_v_3':SJ_a_v_3,
                'SJ_a_v_4':SJ_a_v_4,
                'SJ_a_v_5':SJ_a_v_5,
                'SJ_a_v_6':SJ_a_v_6,
                'SJ_a_v_7':SJ_a_v_7,
                'SJ_a_v_8':SJ_a_v_8,
                'SJ_a_v_9':SJ_a_v_9,
                'SJ_a_v_10':SJ_a_v_10,
                'SJ_a_v_11':SJ_a_v_11,
                'SJ_a_v_12':SJ_a_v_12,
                'SJ_a_v_13':SJ_a_v_13,
                'SJ_a_v_14':SJ_a_v_14,
                'SJ_a_v_15':SJ_a_v_15,
                
                'SJ_a_v_16':SJ_a_v_16,
                'SJ_a_v_17':SJ_a_v_17,
                'SJ_a_v_18':SJ_a_v_18,
                'SJ_a_v_19':SJ_a_v_19,
                'SJ_a_v_20':SJ_a_v_20,
                'SJ_a_v_21':SJ_a_v_21,
                'SJ_a_v_22':SJ_a_v_22,
                'SJ_a_v_23':SJ_a_v_23,
                'trainer_barchart_data':trainer_barchart_data,
                'user_barchart_data':user_barchart_data
                }
            #==================================================================
            
        else:
            op_range="(가동범위 부족)"
            score="(측정 불가)"
            angle_variation = ""
            direct_variation = ""
            feedback="가동범위가 부족합니다."
            trainer_result=""
            user_result=""
            trainer_barchart_data=""
            user_barchart_data=""
        
            context = {
                'exercise': exercise,
                'op_range': op_range,
                'score': score,
                'user_video': user_video,
                'trainer_video':trainer_video,
                'angle_variation': angle_variation,
                'direct_variation': direct_variation,
                'feedback' : feedback,
                
                'SJ_a_v_0':0,
                'SJ_a_v_1':0,
                'SJ_a_v_2':0,
                'SJ_a_v_3':0,
                'SJ_a_v_4':0,
                'SJ_a_v_5':0,
                'SJ_a_v_6':0,
                'SJ_a_v_7':0,
                'SJ_a_v_8':0,
                'SJ_a_v_9':0,
                'SJ_a_v_10':0,
                'SJ_a_v_11':0,
                'SJ_a_v_12':0,
                'SJ_a_v_13':0,
                'SJ_a_v_14':0,
                'SJ_a_v_15':0,
                
                'SJ_a_v_16':0,
                'SJ_a_v_17':0,
                'SJ_a_v_18':0,
                'SJ_a_v_19':0,
                'SJ_a_v_20':0,
                'SJ_a_v_21':0,
                'SJ_a_v_22':0,
                'SJ_a_v_23':0,
                'trainer_barchart_data':trainer_barchart_data,
                'user_barchart_data':user_barchart_data
                }
            
    elif str(request.session.get('exercise_type', False))=="풀업":#풀업
        trainer_video='/static/trainer/pullup_trainer_result.mp4'
        pullup_result_r, pullup_result_l=backend.range_motion(exercise,str(request.user))#등기준촬영
        op_range="왼쪽: "+str(pullup_result_l)+", 오른쪽: "+str(pullup_result_r)
        if 80<=pullup_result_l<120 and 80<=pullup_result_r<120:
            op_range= backend.range_motion(exercise,str(request.user))
            score,angle_variation,feedback=backend.angle_change(exercise,str(request.user))
            trainer_result,user_result,trainer_barchart_part,trainer_barchart_data,user_barchart_part,user_barchart_data=backend.direct_detail(exercise,str(request.user))
            direct_variation="\n<풀업>-방향 변화, 방향 변화점 수\n트레이너\n"+trainer_result+"\n유저\n"+user_result
            
            #승준==============================================================
            
            SJ_a_v_0=backend.SJ_angle_change(exercise,str(request.user))[0]
            SJ_a_v_1=backend.SJ_angle_change(exercise,str(request.user))[1]
            SJ_a_v_2=backend.SJ_angle_change(exercise,str(request.user))[2]
            SJ_a_v_3=backend.SJ_angle_change(exercise,str(request.user))[3]
            SJ_a_v_4=backend.SJ_angle_change(exercise,str(request.user))[4]
            SJ_a_v_5=backend.SJ_angle_change(exercise,str(request.user))[5]
            SJ_a_v_6=backend.SJ_angle_change(exercise,str(request.user))[6]
            SJ_a_v_7=backend.SJ_angle_change(exercise,str(request.user))[7]
            SJ_a_v_8=backend.SJ_angle_change(exercise,str(request.user))[8]
            SJ_a_v_9=backend.SJ_angle_change(exercise,str(request.user))[9]
            SJ_a_v_10=backend.SJ_angle_change(exercise,str(request.user))[10]
            SJ_a_v_11=backend.SJ_angle_change(exercise,str(request.user))[11]
            SJ_a_v_12=backend.SJ_angle_change(exercise,str(request.user))[12]
            SJ_a_v_13=backend.SJ_angle_change(exercise,str(request.user))[13]
            SJ_a_v_14=backend.SJ_angle_change(exercise,str(request.user))[14]
            SJ_a_v_15=backend.SJ_angle_change(exercise,str(request.user))[15]
            
            SJ_a_v_16=0
            SJ_a_v_17=0
            SJ_a_v_18=0
            SJ_a_v_19=0
            SJ_a_v_20=0
            SJ_a_v_21=0
            SJ_a_v_22=0
            SJ_a_v_23=0
            
            context = {
                'exercise': exercise,
                'op_range': op_range,
                'score': score,
                'user_video': user_video,
                'trainer_video':trainer_video,
                'angle_variation': angle_variation,
                'direct_variation': direct_variation,
                'feedback' : feedback,
                
                'SJ_a_v_0':SJ_a_v_0,
                'SJ_a_v_1':SJ_a_v_1,
                'SJ_a_v_2':SJ_a_v_2,
                'SJ_a_v_3':SJ_a_v_3,
                'SJ_a_v_4':SJ_a_v_4,
                'SJ_a_v_5':SJ_a_v_5,
                'SJ_a_v_6':SJ_a_v_6,
                'SJ_a_v_7':SJ_a_v_7,
                'SJ_a_v_8':SJ_a_v_8,
                'SJ_a_v_9':SJ_a_v_9,
                'SJ_a_v_10':SJ_a_v_10,
                'SJ_a_v_11':SJ_a_v_11,
                'SJ_a_v_12':SJ_a_v_12,
                'SJ_a_v_13':SJ_a_v_13,
                'SJ_a_v_14':SJ_a_v_14,
                'SJ_a_v_15':SJ_a_v_15,
                
                'SJ_a_v_16':SJ_a_v_16,
                'SJ_a_v_17':SJ_a_v_17,
                'SJ_a_v_18':SJ_a_v_18,
                'SJ_a_v_19':SJ_a_v_19,
                'SJ_a_v_20':SJ_a_v_20,
                'SJ_a_v_21':SJ_a_v_21,
                'SJ_a_v_22':SJ_a_v_22,
                'SJ_a_v_23':SJ_a_v_23,
                'trainer_barchart_data':trainer_barchart_data,
                'user_barchart_data':user_barchart_data
                }
            
            #==================================================================
            
        else:
            op_range="(가동범위 부족)"
            score="(측정 불가)"
            angle_variation = ""
            direct_variation = ""
            feedback="가동범위가 부족합니다."
            trainer_result=""
            user_result=""
            trainer_barchart_data=""
            user_barchart_data=""
    
            context = {
                'exercise': exercise,
                'op_range': op_range,
                'score': score,
                'user_video': user_video,
                'trainer_video':trainer_video,
                'angle_variation': angle_variation,
                'direct_variation': direct_variation,
                'feedback' : feedback,
                
                'SJ_a_v_0':0,
                'SJ_a_v_1':0,
                'SJ_a_v_2':0,
                'SJ_a_v_3':0,
                'SJ_a_v_4':0,
                'SJ_a_v_5':0,
                'SJ_a_v_6':0,
                'SJ_a_v_7':0,
                'SJ_a_v_8':0,
                'SJ_a_v_9':0,
                'SJ_a_v_10':0,
                'SJ_a_v_11':0,
                'SJ_a_v_12':0,
                'SJ_a_v_13':0,
                'SJ_a_v_14':0,
                'SJ_a_v_15':0,
                
                'SJ_a_v_16':0,
                'SJ_a_v_17':0,
                'SJ_a_v_18':0,
                'SJ_a_v_19':0,
                'SJ_a_v_20':0,
                'SJ_a_v_21':0,
                'SJ_a_v_22':0,
                'SJ_a_v_23':0,
                'trainer_barchart_data':trainer_barchart_data,
                'user_barchart_data':user_barchart_data
                }

    
    #print(trainer_barchart_part,trainer_barchart_data,user_barchart_part,user_barchart_data)
    return render(request, "users/feedback.html", context)



def time(request):
    val=models.Document.objects.get(id=request.user).time
    if val != None:
        if val>=95:
            return redirect("filecomplete/")
        
        
    context={
        "time":str(val)+"%"
    }    
    
    return render(request, "users/time.html",context)
