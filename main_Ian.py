from Iris import Iris
from SoyBean import SoyBean as SB
from BreastCancer import BreastCancer as BC
from Glass import Glass
from Vote import Vote
import pandas as pd

def merge(*dfs):    
    merge_df = pd.concat(list(dfs)).reset_index(drop=True)
    merge_df.to_csv("merged.csv")
    summary_df = merge_df.groupby(by=['Bin_Number', 'M_Value', 'Prob_Value'])[['Average']].agg('mean')
    summary_df.to_csv("summary.csv")
    # print(summary_df.loc[summary_df['Average'] == summary_df['Average'].max()])

    tuningdf = summary_df.loc[summary_df['Average'] == summary_df['Average'].max()]
    tuningdf.to_csv("tuningdf.csv")
                    



if __name__ == '__main__':
    tuningdf = pd.DataFrame
    starting_bin = 1
    tuning = 3
    if(tuningdf['Average'].iloc[0] != None):
        print(tuningdf['Bin_Number'].iloc[0])
        print(tuningdf['M_Value'].iloc[0])

    I = Iris()
    I.test(tuning, starting_bin)
    S = SB()
    S.test(tuning, starting_bin)
    C = BC()
    C.test(tuning, starting_bin)
    G = Glass()
    G.test(tuning, starting_bin)
    V = Vote()
    V.test(tuning, starting_bin)
    merge(I.analysis_df, S.analysis_df, C.analysis_df, G.analysis_df, V.analysis_df)

