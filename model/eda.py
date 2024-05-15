import pandas as pd
from lib import outliers as out, summary as sum, graphic as gra

pd.set_option('display.max_column', None)
pd.set_option('display.width', 5000)

df = pd.read_csv('datasets/dataset.csv')

df.shape

df.nunique()

df.head()

result = out.grab_col_names(df)

cat_cols, num_cols = result[0], result[1]

gra.plot_numerical_col(df, num_cols=num_cols, plot_type='kde')

gra.plot_categoric_col(df, cat_cols)

sum.check_df(df)

sum.cat_summary(df, cat_cols)

for col in num_cols:
    sum.target_summary_with_num(df, 'review_score', numerical_col=col)

for col in cat_cols:
    sum.target_summary_with_cat(df, 'review_score', categorical_col=col)

sum.rare_analyser(df, 'review_score', cat_cols)

sum.correlation_matrix(df, num_cols)

sum.missing_values_table(df, num_cols)

out.for_check(df, num_cols)

