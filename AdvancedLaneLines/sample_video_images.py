from moviepy.editor import VideoFileClip
import cv2
import numpy as np

for f in ['project_video', 'challenge_video', 'harder_challenge_video']:
    clip = VideoFileClip(f + '.mp4')
    step = clip.duration / 10
    i = 0
    count = 0
    while i < clip.duration:
        img = clip.get_frame(i)
        cv2.imwrite('video_images/{0}_{1}.jpg'.format(f,count), img[:,:,::-1])
        i += step
        count += 1


