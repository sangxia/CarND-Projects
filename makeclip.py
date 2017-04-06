import pandas as pd
from moviepy import editor as mpy

log_pathname = 'data-track2/'
log_filename = log_pathname + 'driving_log.csv'
has_header = False
is_relative = False
if has_header:
    df = pd.read_csv(log_filename)
else:
    df = pd.read_csv(log_filename, header=None, \
            names=['center','left','right','steering','throttle','brake','speed'])
if not is_relative:
    for col in ['center','left','right']:
        df.loc[:,col] = df[col].apply(lambda s: s[s.index('IMG'):])

filelist = list(log_pathname + df['center'])
clip = mpy.ImageSequenceClip(filelist, fps=24)
clip.write_videofile('sample-2.mp4')

