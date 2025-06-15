import queue
import threading

from april import vision_loop
from mujoco_loop import sim_loop

if __name__ == "__main__":
    quueue = queue.Queue()
    pose_q = queue.Queue(maxsize=1)
    threading.Thread(target=vision_loop, args=(pose_q,), daemon=True).start()

    sim_loop(pose_q)
