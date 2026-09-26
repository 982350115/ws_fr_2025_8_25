#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from example_interfaces.srv import SetBool  # 使用标准的布尔服务 (True=Open, False=Close)

# 导入您刚才重命名的SDK模块
from gripper_driver.actuator_sdk import ActuatorSDK

class GripperServiceNode(Node):
    def __init__(self):
        super().__init__('gripper_service_node')

        # --- 配置参数 ---
        self.declare_parameter('port', '/dev/ttyUSB0')
        self.declare_parameter('baudrate', 115200)
        
        port = self.get_parameter('port').get_parameter_value().string_value
        baud = self.get_parameter('baudrate').get_parameter_value().integer_value

        self.get_logger().info(f'正在连接夹爪: {port} @ {baud}')

        # --- 初始化 SDK ---
        try:
            # 实例化 SDK (Slave ID 默认 1)
            self.sdk = ActuatorSDK(port, slave_id=1, baudrate=baud, timeout=0.5)
            # 使能电机
            self.sdk.enable(True)
            self.get_logger().info('夹爪连接成功且已使能！')
        except Exception as e:
            self.get_logger().error(f'夹爪连接失败: {e}')
            # 这里的 exit 可能会导致节点直接挂掉，实际部署需谨慎
            raise e

        # --- 创建服务 ---
        # 服务名: /gripper/set_open
        # 类型: SetBool (data: True/False)
        self.srv = self.create_service(SetBool, 'gripper_set_open', self.handle_gripper_command)
        self.get_logger().info('服务已就绪: /gripper/set_open')

    def handle_gripper_command(self, request, response):
        """
        处理服务请求
        request.data = True  -> 张开 (Target = 2000, 可修改)
        request.data = False -> 闭合 (Target = 0, 可修改)
        """
        target_pos = 0 if request.data else 2000
        action_name = "张开" if request.data else "闭合"

        self.get_logger().info(f'收到指令: {action_name} (目标位置: {target_pos})')

        try:
            # 1. 发送运动指令
            # 参数: pos, speed(%), force(%), accel, decel, trigger
            self.sdk.temp_move(position_mm=target_pos, speed_pct=100, force_pct=50, trigger=True)
            
            # 2. 阻塞等待动作完成 (使用 SDK 自带的等待函数)
            # 注意：在简单的 Service 中阻塞是允许的，但如果动作非常慢，建议改用 Action
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
                self.get_logger().info(f'{action_name} 完成. 位置: {current_pos}')

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