from Iris import Iris
from SoyBean import SoyBean as SB
from BreastCancer import BreastCancer as BC
from Glass import Glass
from Vote import Vote
import pandas as pd

def merge(i,s,c,g,v):
    print(i.groupby(by=['Bin_Number', 'M_Value', 'Prob_Value'])[['Average']].agg('mean'))
    s.groupby(by=['Bin_Number', 'M_Value', 'Prob_Value'])[['Average']].agg('mean')
    c.groupby(by=['Bin_Number', 'M_Value', 'Prob_Value'])[['Average']].agg('mean')
    g.groupby(by=['Bin_Number', 'M_Value', 'Prob_Value'])[['Average']].agg('mean')
    v.groupby(by=['Bin_Number', 'M_Value', 'Prob_Value'])[['Average']].agg('mean')
    
    # merge_df = pd.concat(list(dfs)).reset_index(drop=True)
    # merge_df.to_csv("merged.csv")
    # summary_df = merge_df.groupby(by=['Bin_Number', 'M_Value', 'Prob_Value'])[['Average']].agg('mean')
    # summary_df.to_csv("summary.csv")
    # print(summary_df.loc[summary_df['Average'] == summary_df['Average'].max()])



if __name__ == '__main__':
    starting_bin = 3
    tuning = 2
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

