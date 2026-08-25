注意事项：
    1. 本工作空间下各包功能：  demo_fairino16：可以实现基于导纳控制的机械臂重力补偿。
                           fairino_description： 里面包括法奥机器人的较为常见的机械臂urdf模型以及相配套的mesh文件。
                           fairino_hardware: 用于与实体机械臂建立通信。若要连接实体机械臂，需在使用前将相应xxx.ros2_control.xacro文件中：<ros2_control name="${name}" type="system">
                                                                                                                                  <hardware>
                                                                                                                                      <plugin>mock_components/GenericSystem</plugin>
                                                                                                                                  </hardware>
                                                                                                                        改为：<ros2_control name="${name}" type="system">
                                                                                                                                  <hardware>
                                                                                                                                      <plugin>fairino_hardware/FairinoHardwareInterface</plugin>
                                                                                                                                  </hardware>    
                                                                                                                        随后至少编译目标包与fairino_hardware，刷新环境，即可将电脑rviz与实体相连。
                           fairino_msgs:与fairino_hardware类似，需提前编译。
                           fairino16_ctpm2f20_1：我的主要包  其中包含视觉抓取、视觉双杯倒水等demo。
                           fairino16_v6_moveit2_config：这个包中主要不包含夹爪、相机等为纯粹的机械臂模型。有一个末端走圆形轨迹的demo。
                           gripper_driver：夹爪驱动程序。本包中夹爪驱动逻辑为使用py代码通过sdk控制夹爪开闭，与ros、moveit不同，不走moveit的那一套消息传递。
                           2025_12:其中包含与前文所说的抓取、倒水等demo配套的视觉程序。
                           Camera_hand-eye_calibration：存放orbbec相机的手眼标定程序。01、02文件两步走：01负责收集数据、02负责计算、calc文件负责修正。
                           Trajectory_Extraction：其中包含视觉提取轨迹、并将轨迹发布至话题中的文件。