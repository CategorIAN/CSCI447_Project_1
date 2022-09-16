from Iris import Iris
from SoyBean import SoyBean as SB
from BreastCancer import BreastCancer as BC
from Glass import Glass
from Vote import Vote
import pandas as pd
from os.path import exists

def merge(*dfs):    
    merge_df = pd.concat(list(dfs)).reset_index(drop=True)
    merge_df.to_csv("merged.csv")
    summary_df = merge_df.groupby(by=['Bin_Number', 'M_Value', 'Prob_Value'])[['Average']].agg('mean')
    summary_df.to_csv("summary.csv")
    # print(summary_df.loc[summary_df['Average'] == summary_df['Average'].max()])

    tuningdf = summary_df.loc[summary_df['Average'] == summary_df['Average'].max()]
    tuningdf.to_csv("tuningdf.csv")
                    



if __name__ == '__main__':
    
    tuning = 5
    if(exists("tuningdf.csv")):
        tuningdf = pd.read_csv("tuningdf.csv")       
        starting_bin = tuningdf['Bin_Number'].iloc[0]
        m_val = tuningdf['M_Value'].iloc[0]
    else:
        starting_bin = 1
        m_val = 1
    

    I = Iris()
    I.test(tuning, starting_bin, m_val)
    S = SB()
    S.test(tuning, starting_bin, m_val)
    C = BC()
    C.test(tuning, starting_bin, m_val)
    G = Glass()
    G.test(tuning, starting_bin, m_val)
    V = Vote()
    V.test(tuning, starting_bin, m_val)
    merge(I.analysis_df, S.analysis_df, C.analysis_df, G.analysis_df, V.analysis_df)

