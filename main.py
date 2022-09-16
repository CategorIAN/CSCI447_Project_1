from Iris import Iris
from SoyBean import SoyBean as SB
from BreastCancer import BreastCancer as BC
from Glass import Glass
from Vote import Vote
import pandas as pd
from os.path import exists

#merge all the analysis datasets into one and find the best combination for hyperparameters
def merge(*dfs):    
    #merge
    merge_df = pd.concat(list(dfs)).reset_index(drop=True)
    merge_df.to_csv("merged.csv")

    #create summary dataframe that avarges each of the combinations and finds the avarges for each combination
    summary_df = merge_df.groupby(by=['Bin_Number', 'M_Value', 'Prob_Value'])[['Average']].agg('mean')
    summary_df.to_csv("summary.csv")

    #find the best combination of hyperparameters
    tuningdf = summary_df.loc[summary_df['Average'] == summary_df['Average'].max()]
    tuningdf.to_csv("tuningdf.csv")
                    
    # merge_df.to_latex("merge_latex.txt")
    # tuningdf.to_latex("tuning_latex.txt")
    summary_df.to_latex("summary_latex.txt")


if __name__ == '__main__':
    
    tuning = 10
    #if there is a tuningdf then some tuning has already been done so start from the found hyperparameters
    if(exists("tuningdf.csv")):
        tuningdf = pd.read_csv("tuningdf.csv")       #read csv
        starting_bin = tuningdf['Bin_Number'].iloc[0]   #best bin number
        m_val = tuningdf['M_Value'].iloc[0] #best m_val

    #otherwise have starting values
    else:
        starting_bin = 1
        m_val = 1
    
    #got through and test each data set
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

    #merge the analysis dataframes
    merge(I.analysis_df, S.analysis_df, C.analysis_df, G.analysis_df, V.analysis_df)
    
    # I.analysis_df.to_latex("i_analysis_latex.txt")

