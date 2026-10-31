# 同一帧底板与齿轮检测结果

- `rgb.png`：不含标注的检测输入彩色图片；已经相机去畸变，并非未经处理的传感器像素。
- `rgbd.npz`：同一帧 RGB、对齐深度（米）和内参 K。
- `poses_camera.json`：底板与红色 24 齿齿轮相对相机的完整 4×4 变换、XYZ（米/毫米）、四元数（wxyz）、帧编号及质量指标。
- `T_camera_plate.txt`、`T_camera_gear.txt`：便于直接读取的 4×4 矩阵。
- `plate_result.json`：对这一帧重新运行 FoundationPose 的底板结果，可加载到 Viser。
- `result.json`、`object_in_camera.txt`：保存这一帧时实时齿轮跟踪器的原始输出。
- `overlay.png`：齿轮检测标注；`plate_overlay.png`：底板轮廓；`detections_overlay.png`：底板轮廓及两物体坐标轴。
- `scene.ply`：这一帧的场景点云。
- `handeye_result.json`：本次运行使用的手眼标定文件备份；相机坐标结果本身没有乘世界外参。

相机坐标：+X 向右、+Y 向下、+Z 向前。矩阵满足 p_camera = T_camera_object @ p_object，平移单位为米。物体坐标系沿用仿真 CAD 网格原点，不能将底板原点和齿轮原点等同于物体底面。

底板与齿轮来自同一帧 RGB-D。底板使用离线单帧配准，齿轮使用该帧保存的在线跟踪结果。齿轮有齿距旋转对称性，绕自身轴的朝向不具有唯一性。
