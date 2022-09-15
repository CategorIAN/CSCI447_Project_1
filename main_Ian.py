from Iris import Iris
from SoyBean import SoyBean as SB
from BreastCancer import BreastCancer as BC
from Glass import Glass
from Vote import Vote
import pandas as pd

def merge(*dfs):
    merge_df = pd.concat(list(dfs)).reset_index(drop=True)
    merge_df.to_csv("merged.csv")
    summary_df = merge_df.groupby(by=['Bin_Number'])[['Average']].agg('mean')
    summary_df.to_csv("summary.csv")
    print(summary_df.loc[summary_df['Average'] == summary_df['Average'].max()])



if __name__ == '__main__':
    I = Iris()
    I.test()
    S = SB()
    S.test()
    C = BC()
    C.test()
    G = Glass()
    G.test()
    V = Vote()
    V.test()
    merge(I.analysis_df, S.analysis_df, C.analysis_df, G.analysis_df, V.analysis_df)

