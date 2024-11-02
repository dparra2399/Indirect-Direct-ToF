import pandas as pd

df = pd.read_csv('deap_q_raw.csv')

df_2 = pd.read_csv('deapq_output_2_27.csv')

prof_df = pd.read_csv('prolific_export_6509d457efda6f6705b43312.csv')
prof_df = prof_df[['Submission id', 'Sex']]

unique_df = df.drop_duplicates(subset='Participant Private ID')
unique_df = unique_df[['Participant Private ID', 'Participant External Session ID']]

merged_deep_df = pd.merge(unique_df, df_2, left_on='Participant Private ID', right_on='participant_private_id', how='inner')
merged_deep_df = merged_deep_df.drop(columns=['Participant Private ID', 'submission_id'])

merged_prof_df = pd.merge(merged_deep_df, prof_df, left_on='Participant External Session ID', right_on='Submission id', how='inner')
merged_prof_df = merged_prof_df.drop(columns=['Participant External Session ID'])

column_to_move = merged_prof_df.pop('Submission id')
merged_prof_df.insert(0, 'submission_id', column_to_move)

merged_prof_df = merged_prof_df.drop(columns=['gender'])
column_to_move = merged_prof_df.pop('Sex')
merged_prof_df.insert(3, 'Sex', column_to_move)


merged_prof_df.to_csv('merged.csv', index=False)
print(len(merged_prof_df))
print(len(prof_df))
print(len(unique_df))
print(len(df_2))