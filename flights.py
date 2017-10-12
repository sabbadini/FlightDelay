#--> https://gist.github.com/martinwicke/6838c23abdc53e6bcda36ed9f40cff39

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# We're using pandas to read the CSV file. This is easy for small datasets, but for large and complex datasets,
# tensorflow parsing and processing functions are more powerful.
import pandas as pd
import numpy as np

# The CSV file does not have a header, so we have to fill in column names.
names = [
    'delayweathersum', 
    'carrier', 
    'meantemptorigin', 
    'precipinorigin', 
    'snowinorigin',
    'gustmphorigin',
    'visbilitymilesorigin',
    'rainorigin',
    'thundershowersorigin',
    'snoworigin',
    'fogorigin',
    'meantempfdestination',
    'precipindestination',
    'snowindestination',
    'gustmphdestination',
    'visbilitymilesdestination',
    'raindestination',
    'thundershowersdestination',
    'snowdestination',
    'fogdestination',
]

# We also have to specify dtypes.
dtypes = {
    'delayweathersum': np.float32, 
    'carrier': str, 
    'meantemptorigin': np.float32, 
    'precipinorigin': np.float32, 
    'snowinorigin': np.float32, 
    'gustmphorigin': np.float32, 
    'visbilitymilesorigin': np.float32, 
    'rainorigin': str, 
    'thundershowersorigin': str, 
    'snoworigin': str, 
    'fogorigin': str, 
    'meantempfdestination': np.float32, 
    'precipindestination': np.float32, 
    'snowindestination': np.float32, 
    'gustmphdestination': np.float32, 
    'visbilitymilesdestination': np.float32, 
    'raindestination': str, 
    'thundershowersdestination': str, 
    'snowdestination': str, 
    'fogdestination': str, 
}

# Read the file.
df = pd.read_csv('2017JanDelayCHI_ATL.csv', names=names, dtype=dtypes, na_values='?')

# Split the data into a training set and an eval set.
training_data = df[:465]
eval_data = df[465:]
test_data = df[:10]

# Separate input features from labels
training_label = training_data.pop('delayweathersum')
eval_label = eval_data.pop('delayweathersum')
test_label = test_data.pop('delayweathersum')

# Now we can start using some TensorFlow.
import tensorflow as tf
print('please make sure that version >= 1.2:')
print(tf.__version__)

# Make input function for training: 
#   num_epochs=None -> will cycle through input data forever
#   shuffle=True -> randomize order of input data
training_input_fn = tf.estimator.inputs.pandas_input_fn(x=training_data, y=training_label, batch_size=64, shuffle=True, num_epochs=None)

# Make input function for evaluation:
#   shuffle=False -> do not randomize input data
eval_input_fn = tf.estimator.inputs.pandas_input_fn(x=eval_data, y=eval_label, batch_size=64, shuffle=False)

#test / predict
#   shuffle=False -> do not randomize input data
test_input_fn = tf.estimator.inputs.pandas_input_fn(x=test_data, y=test_label, batch_size=10, shuffle=False)

# Describe how the model should interpret the inputs. The names of the feature columns have to match the names
# of the series in the dataframe.

#delay_weather_sum = tf.feature_column.numeric_column('delayweathersum')
carrier = tf.feature_column.categorical_column_with_vocabulary_list('carrier', vocabulary_list=['WN', 'OO', 'NK', 'AA', 'DL', 'UA'])
mean_temp_t_origin = tf.feature_column.numeric_column('meantemptorigin')
precip_in_origin = tf.feature_column.numeric_column('precipinorigin')
snow_in_origin = tf.feature_column.numeric_column('snowinorigin')
gust_mph_origin = tf.feature_column.numeric_column('gustmphorigin')
visbility_miles_origin = tf.feature_column.numeric_column('visbilitymilesorigin')
rain_origin = tf.feature_column.categorical_column_with_vocabulary_list('rainorigin', vocabulary_list=['TRUE', 'FALSE'])
thundershowers_origin = tf.feature_column.categorical_column_with_vocabulary_list('thundershowersorigin', vocabulary_list=['TRUE', 'FALSE'])
snow_origin = tf.feature_column.categorical_column_with_vocabulary_list('snoworigin', vocabulary_list=['TRUE', 'FALSE'])
fog_origin = tf.feature_column.categorical_column_with_vocabulary_list('fogorigin', vocabulary_list=['TRUE', 'FALSE'])
mean_temp_f_destination = tf.feature_column.numeric_column('meantempfdestination')
precip_in_destination = tf.feature_column.numeric_column('precipindestination')
snow_in_destination = tf.feature_column.numeric_column('snowindestination')
gust_mph_destination = tf.feature_column.numeric_column('gustmphdestination')
visbility_miles_destination = tf.feature_column.numeric_column('visbilitymilesdestination')
rain_destination = tf.feature_column.categorical_column_with_vocabulary_list('raindestination', vocabulary_list=['TRUE', 'FALSE'])
thundershowers_destination = tf.feature_column.categorical_column_with_vocabulary_list('thundershowersdestination', vocabulary_list=['TRUE', 'FALSE'])
snow_destination = tf.feature_column.categorical_column_with_vocabulary_list('snowdestination', vocabulary_list=['TRUE', 'FALSE'])
fog_destination = tf.feature_column.categorical_column_with_vocabulary_list('fogdestination', vocabulary_list=['TRUE', 'FALSE'])

#Linear Regressor

linear_features = [carrier, mean_temp_t_origin, precip_in_origin, snow_in_origin, gust_mph_origin, visbility_miles_origin, rain_origin, thundershowers_origin, snow_origin, fog_origin, mean_temp_f_destination, precip_in_destination, snow_in_destination, gust_mph_destination, visbility_miles_destination, rain_destination, thundershowers_destination, snow_destination, fog_destination]
regressor = tf.contrib.learn.LinearRegressor(feature_columns=linear_features)
regressor.fit(input_fn=training_input_fn, steps=10000)
regressor.evaluate(input_fn=eval_input_fn)

#Deep Neural Network

dnn_features = [
    #numerical features
    mean_temp_t_origin, precip_in_origin, snow_in_origin, gust_mph_origin, visbility_miles_origin,
    mean_temp_f_destination, precip_in_destination, snow_in_destination, gust_mph_destination, visbility_miles_destination, 
    # densify categorical features:
    tf.feature_column.indicator_column(carrier),
    tf.feature_column.indicator_column(rain_origin),
    tf.feature_column.indicator_column(thundershowers_origin),
    tf.feature_column.indicator_column(snow_origin),
    tf.feature_column.indicator_column(fog_origin),
    tf.feature_column.indicator_column(rain_destination), 
    tf.feature_column.indicator_column(thundershowers_destination),
    tf.feature_column.indicator_column(snow_destination),
    tf.feature_column.indicator_column(fog_destination),
]

dnnregressor = tf.contrib.learn.DNNRegressor(feature_columns=dnn_features, hidden_units=[50, 30, 10])
dnnregressor.fit(input_fn=training_input_fn, steps=10000)
dnnregressor.evaluate(input_fn=eval_input_fn)

#Predict

predictions = list(dnnregressor.predict_scores(input_fn=test_input_fn))
print(predictions)

predictionsLarge = list(dnnregressor.predict_scores(input_fn=eval_input_fn))
print(predictionsLarge)

predictionsLinear = list(regressor.predict_scores(input_fn=test_input_fn))
print(predictionsLinear)

predictionsLinearLarge = list(regressor.predict_scores(input_fn=eval_input_fn))
print(predictionsLinearLarge)
