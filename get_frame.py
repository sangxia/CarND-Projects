from moviepy.editor import VideoFileClip
import cv2
import numpy as np

f = 'project_video.mp4'
clip = VideoFileClip(f)
ts = 39.78
fout = 'video_images/debug_2.jpg'
img = clip.get_frame(ts)
cv2.imwrite(fout, img[:,:,::-1])

