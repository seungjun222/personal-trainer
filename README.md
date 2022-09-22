# personal-trainer
사용자의 운동 영상과 트레이너의 운동영상을 비교하여 일치율을 출력하는 프로그램입니다.

웹에서 분석대시보드 형태로 서비스를 제공하도록 제작되었습니다.

프로그램 테스트시 http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/mpi/pose_iter_160000.caffemodel 에서 pose_iter_160000.caffemodel 파일을 다운 받아

model 폴더에 넣어 주세요. 

## 구성도
-백그라운드 로직 수행을 위해 멀티프로세싱이 사용되었습니다.
![image](https://user-images.githubusercontent.com/63800086/174829013-13d70af8-791c-4dc3-94f8-c93348e88ece.png)

## 웹서비스
-서버 컴퓨터 사양에 따라 계산에 많은 시간이 소요될 수 있습니다.(GPU연산을 권장합니다.)
![image](https://user-images.githubusercontent.com/63800086/174832287-59c5e507-da98-4ab4-9e57-0d3048dd7b39.png)
