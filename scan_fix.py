import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan


class ScanFix(Node):
    def __init__(self):
        super().__init__('scan_fix')
        self.n = None
        self.pub = self.create_publisher(LaserScan, '/scan_fixed', 10)
        self.create_subscription(LaserScan, '/scan', self.cb, qos_profile_sensor_data)

    def cb(self, m):
        if self.n is None:
            self.n = int(round((m.angle_max - m.angle_min) / m.angle_increment)) + 1
            self.amin, self.amax, self.inc = m.angle_min, m.angle_max, m.angle_increment
            self.get_logger().info(f'fixed scan size = {self.n}')
        far = m.range_max + 1.0 if math.isfinite(m.range_max) else 100.0
        r = [float(x) if math.isfinite(x) else far for x in list(m.ranges)[:self.n]]
        r += [far] * (self.n - len(r))
        m.ranges = r
        if len(m.intensities) > 0:
            i = [float(x) if math.isfinite(x) else 0.0 for x in list(m.intensities)[:self.n]]
            i += [0.0] * (self.n - len(i))
            m.intensities = i
        m.angle_min, m.angle_max, m.angle_increment = self.amin, self.amax, self.inc
        self.pub.publish(m)


def main():
    rclpy.init()
    rclpy.spin(ScanFix())


main()
