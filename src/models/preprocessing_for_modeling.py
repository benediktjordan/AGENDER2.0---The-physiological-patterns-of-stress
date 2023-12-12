

#my splitting function
def split_ensemble(df, split_percentage):
	test_indices = random.sample(range(0, df.shape[0]),
								 int(split_percentage*df.shape[0]))
	df_test = pd.DataFrame()
	df_train = df.copy()

	df_train = df_train.drop(test_indices, axis = 0)
	for i in test_indices:
		df_test = df_test.append(df.iloc[i])

	return df_train, df_test
