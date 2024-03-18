# file to combine files together
import pandas as pd


df1 = pd.read_csv("data/cartpole_data.csv")
df2 = pd.read_csv("data/augmented_data_adv.csv")
df3 = pd.read_csv("data/augmented_data_gaussian.csv")
df4 = pd.read_csv("data/augmented_data_uniform.csv")
df5 = pd.read_csv("data/gauss_adv.csv")
df6 = pd.read_csv("data/uniform_adv.csv")


df_adv = pd.concat([df1, df2])
df_gauss = pd.concat([df1, df3])
df_uniform = pd.concat([df1, df4])
df_adv_gauss = pd.concat([df1, df5])
df_adv_uniform = pd.concat([df1, df6])

c_adv_gauss = pd.concat([df1, df2, df3])
c_adv_uniform = pd.concat([df1, df2, df4])
c_gauss_uniform = pd.concat([df1, df2, df4])
c_all = pd.concat([df1, df2, df3, df4])

# singular augmented files
df_adv.to_csv("data/combined_adv.csv", index=False)
df_gauss.to_csv("data/combined_gauss.csv", index=False)
df_uniform.to_csv("data/combined_uniform.csv", index=False)
# stacked augmented files
df_adv_gauss.to_csv("data/combined_adv_gauss.csv", index=False)
df_adv_uniform.to_csv("data/combined_adv_uniform.csv", index=False)

# combined data augmented files
c_adv_gauss.to_csv("data/add_adv_gauss.csv", index=False)
c_adv_uniform.to_csv("data/add_adv_uniform.csv", index=False)
c_gauss_uniform.to_csv("data/add_gauss_uniform.csv", index=False)
c_all.to_csv("data/add_all.csv", index=False)
