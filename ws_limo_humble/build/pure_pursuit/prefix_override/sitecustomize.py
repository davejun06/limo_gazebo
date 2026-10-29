import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/davejun06/LIMO_GAZEBO/ws_limo_humble/install/pure_pursuit'
