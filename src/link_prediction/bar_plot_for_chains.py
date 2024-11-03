import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
# Function to calculate outlier percentage for each column
def calculate_outlier_percentage(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = column[(column < lower_bound) | (column > upper_bound)]
    return (len(outliers) / len(column)) * 100


# Generate synthetic data (replace this part with loading real TSV files)
file1 = sys.argv[1]
file2 = sys.argv[2]
threshold = int(sys.argv[3])


df1 = pd.read_csv(file1, sep='\t', header=None, names=['Train', 'Test', 'Inferred'])
df2 = pd.read_csv(file2, sep='\t', header=None, names=['Train', 'Test', 'Inferred'])

     
filtered_df1 = df1[df1['Train'] < threshold]
filtered_df2 = df2[df2['Train'] < threshold]

common_indices = filtered_df1.index.intersection(filtered_df2.index)

df1 = df1.loc[common_indices]
df2 = df2.loc[common_indices]

df1 = np.log(df1)
df2 = np.log(df2)



print(df1)
print(df2)

combined_data = [df1['Train'], df1['Test'], df1['Inferred'],
                 df2['Train'], df2['Test'], df2['Inferred']]



# Apply function to each column in the DataFrame
outlier_percentages1 = df1.apply(calculate_outlier_percentage)
outlier_percentages2 = df2.apply(calculate_outlier_percentage)

print(outlier_percentages1)
print(outlier_percentages2)



# Create colored box plot
fig, ax = plt.subplots(figsize=(8, 6))
box = ax.boxplot(combined_data, patch_artist=True, showfliers=True, whis=1.5, showmeans=True, meanline=True)

# Colors for each pair of box plots
box_colors = ['#68A05F', '#5F68A0', '#A05F68']
for i, patch in enumerate(box['boxes']):
    patch.set_facecolor(box_colors[i % 3])  # Alternate colors for each file

# Set plot titles and labels
ax.set_title("Training, Testing, and Inferred Data")
ax.set_xticklabels(["Train (Normal)", "Test (Normal)", "Inferred (Normal)", "Train (Transitive)", "Test (Transitive)", "Inferred (Transitive)"], rotation=45)
ax.set_ylabel("log(Mean Rank)")

# Display the plot
plt.tight_layout()
plt.savefig("box_plot.eps", format='eps', dpi=300)
# plt.show()
