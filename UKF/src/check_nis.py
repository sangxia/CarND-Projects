import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', \
            help="UKF output file containing sensor type and NIS")
    arg = vars(parser.parse_args())
    infile = arg['infile']
    ref = {'radar': (0.352, 7.815), \
            'lidar': (0.103, 5.991) }
    df = pd.read_table(infile)
    sensors = dict(df['sensor_type'].value_counts())
    for s in sensors:
        nis = df.loc[df['sensor_type']==s, 'NIS']
        print(s, 'actual', tuple(nis.quantile([0.05,0.95])), \
                'reference', ref[s])
    
