#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from example_interfaces.srv import SetBool
# --- 新增依赖 ---
from sensor_msgs.msg import JointState 
from gripper_driver.actuator_sdk import ActuatorSDK

class GripperServiceNode(Node):
    def __init__(self):
        super().__init__('gripper_service_node')

        # --- 1. 配置参数 ---
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 115200)
        
        port = self.get_parameter('port').get_parameter_value().string_value
        baud = self.get_parameter('baudrate').get_parameter_value().integer_value

        self.get_logger().info(f'正在连接夹爪: {port} @ {baud}')

        # --- 2. 初始化 SDK ---
        try:
            self.sdk = ActuatorSDK(port, slave_id=1, baudrate=baud, timeout=3.0)
            self.sdk.enable(True)
            self.get_logger().info('夹爪连接成功且已使能！')
        except Exception as e:
            self.get_logger().error(f'夹爪连接失败: {e}')
            raise e

        # --- 3. 创建服务 (用于接收控制指令) ---
        self.srv = self.create_service(SetBool, 'gripper_set_open', self.handle_gripper_command)
        self.get_logger().info('服务已就绪: /gripper_set_open')

        # --- 4. [新增] 创建状态发布者 (让 RViz 看到夹爪动) ---
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        
        # 创建定时器，以 20Hz (0.05秒) 的频率读取硬件并发布状态
        self.timer = self.create_timer(0.05, self.publish_joint_state)

        # ⚠️ 注意：这个名字必须完全等于你 URDF/SRDF 中定义的关节名
        self.joint_name = "gripper_left_joint" 

    def publish_joint_state(self):
        """定时读取硬件位置并发布给 ROS"""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [self.joint_name]

        try:
            # A. 读取硬件原始数值 (假设范围是 0 ~ 2000)
            raw_pos = self.sdk.feedback_position()
            
            # B. [需要微调] 数值映射：将脉冲数转换为米
            # 假设：0 是闭合(0m)，2000 是最大张开(0.025m)
            # 如果你的方向反了（0是张开），请修改公式
            max_pulse = 2000.0
            max_dist_meter = 0.025 # 2.5cm
            
            # 简单的线性变换公式
            current_pos_meter = (raw_pos / max_pulse) * max_dist_meter
            
            # 如果 URDF 里定义的“张开”是负数（例如 -0.01），这里可能需要加负号
            # current_pos_meter = -current_pos_meter 

            msg.position = [current_pos_meter]
            msg.velocity = [] # 暂时不需要
            msg.effort = []   # 暂时不需要
            
            # C. 发布消息 -> RobotStatePublisher -> RViz
            self.joint_pub.publish(msg)
            
        except Exception:
            # 串口偶尔读取失败是正常的，忽略本次错误，不让节点崩溃
            pass

    def handle_gripper_command(self, request, response):
        """处理服务请求"""
        # 根据你的逻辑：True(Open)->0, False(Close)->2000
        # ⚠️ 注意：这里的 0 和 2000 是否和上面的 publish_joint_state 里的方向一致？
        target_pos = 0 if request.data else 2000
        action_name = "张开" if request.data else "闭合"

        self.get_logger().info(f'收到指令: {action_name} (目标: {target_pos})')

        try:
            # 1. 发送运动指令
            self.sdk.temp_move(position_mm=target_pos, speed_pct=100, force_pct=50, trigger=True)
            
            # 2. 阻塞等待
            result = self.sdk.wait_until_pos_or_torque(timeout=5.0)

            # 3. 构建反馈
            if result == 'timeout':
                response.success = False
                response.message = f'{action_name} 超时！'
                self.get_logger().warn(f'{action_name} 动作超时')
            else:
                current_pos = self.sdk.feedback_position()
                response.success = True
                response.message = f'{action_name} 成功. 状态: {result}, 当前位置: {current_pos}'
                self.get_logger().info(f'{action_name} 完成.')

        except Exception as e:
            response.success = False
            response.message = f'驱动异常: {str(e)}'
            self.get_logger().error(f'驱动执行异常: {e}')

        return response

def main(args=None):
    rclpy.init(args=args)
    node = GripperServiceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()