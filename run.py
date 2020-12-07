# Basic required libraries
import urllib.request as request
import json
import os
import pandas as pd
import datetime
import collections
import numpy as np

# ML libraries
from keras.models import Sequential
from keras.layers import Dense
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.utils import shuffle
from keras import callbacks


def validateFeaturizeData(data, user_id, colnames):
    valid_str = 'Be Authentic. Be Yourself. Be Typing.'
    feature_length = len(valid_str) + 1  # 1 for number of backspaces.
    user_feature_df = pd.DataFrame(columns=colnames)
    invalid_count = 0
    invalid_idx = []
    total_count = len(data)
    #     if total_count < 300:
    #         print('insufficient data for user {}'.format(user_id))
    #         return None, ()
    for idx, typed_str in enumerate(data):
        processed_str = []
        prev_time_stamp = None
        backstrokes = 0
        user_features = []
        for char_dict in typed_str:
            curr_char = char_dict["character"]
            if curr_char == "[backspace]":
                processed_str.pop()  # remove the last char
                user_features.pop()  # remove the typing time corresponding to last char
                backstrokes += 1
            else:
                curr_time_stamp = datetime.datetime.strptime(char_dict['typed_at'], '%Y-%m-%dT%H:%M:%S.%f')
                if prev_time_stamp:
                    typing_time = (curr_time_stamp - prev_time_stamp).total_seconds()
                else:
                    typing_time = 0.0
                processed_str.append(curr_char)
                prev_time_stamp = curr_time_stamp
                user_features.append(typing_time)

        filtered_str = ''.join(processed_str)
        if filtered_str == valid_str:
            # append it to the dataframe.
            user_features.append(backstrokes)
            if len(user_features) != feature_length:
                print('Mismatch in feature length for a valid string.\n\
The original string is {}.\n\
The processed string is {}.\n\
The number of backspaces are {}.\n\
User features list is {}'.format(typed_str, processed_str, backstrokes, user_features))

            else:
                user_feature_df = user_feature_df.append(pd.Series(user_features, index=colnames), ignore_index=True)
        else:
            #             print('invalid string: '+filtered_str)# do nothing and pass
            invalid_count += 1
            invalid_idx.append(idx)

    user_feature_df['user_id'] = user_id
    #     print(user_feature_df)
    #     print(user_id, total_count, invalid_count)
    return user_feature_df, (total_count, invalid_count), invalid_idx

def Train(train_url, checkpoint_filepath):

    domain = '/'.join(train_url.split('/')[:-1]) #'https://challenges.unify.id/v1/mle'
    total_users = 0
    next_user = train_url.split('/')[-1] #'user_4a438fdede4e11e9b986acde48001122.json'
    url_error = 0
    stats = collections.defaultdict(list)
    unenrolled_users = []
    train_data = pd.DataFrame(columns=col_names)
    while next_user:
        url = os.path.join(domain, next_user)
        with request.urlopen(url) as response:
            if response.getcode() == 200:
                source = response.read()
                data = json.loads(source)
                total_users += 1
                user_df, user_stats, _ = validateFeaturizeData(data['user_data'], next_user, col_names[:-1])
                if user_stats[0] - user_stats[1] >= 300:
                    stats['user'].append(next_user)
                    stats['total_count'].append(user_stats[0])
                    stats['invalid_count'].append(user_stats[1])
                    train_data = train_data.append(user_df)
                else:
                    # user cannot be enrolled into production due to threshold.
                    unenrolled_users.append(next_user)

                next_user = data['next']
            else:
                print('An error occurred while attempting to retrieve data from the API. Invalid User hash {}'.format(
                    next_user))
                url_error += 1

    # List of enrolled and unenrolled Users
    print('Enrolled users are \n{}'.format(stats['user']))
    print('\nUnenrolled users are \n{}'.format(unenrolled_users))

    no_classes = int(train_data.user_id.nunique())
    train_data = shuffle(train_data)
    X_train = train_data.iloc[:, 1:-1]
    Y_train = train_data.iloc[:, -1]
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y_train)
    encoded_Y = encoder.transform(Y_train)
    mapping = dict(zip(range(len(encoder.classes_)), encoder.classes_))
    mapping[-1] = 'invalid input'
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)

    # define baseline model
    def baseline_model(input_dims, hidden_units, no_classes):
        # create model
        model = Sequential()
        model.add(Dense(hidden_units[0], input_dim=input_dims, activation='relu'))
        model.add(Dense(hidden_units[1], activation='relu'))
        model.add(Dense(no_classes, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    model = baseline_model(no_features, [100, 50], no_classes)
    print(model.summary())
    model_checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    model.fit(X_train, dummy_y, epochs=100, batch_size=1024, validation_split=0.1,
              callbacks=[model_checkpoint_callback])

    # User performance report
    train_pred_class = np.argmax(model.predict(X_train), axis=-1)
    perf_df = pd.DataFrame({'y_true': encoded_Y, 'y_pred': train_pred_class})
    perf_df['outcome'] = perf_df.y_true == perf_df.y_pred
    perf_df = perf_df.groupby('y_true')['outcome'].agg(['sum', 'count'])
    perf_df.reset_index(inplace=True)
    perf_df['acc'] = perf_df['sum'] / perf_df['count']
    perf_df.acc.describe()


    return model, mapping

def Test(test_url, col_names, model, mapping):
    # Test Data
    # test_url = 'https://challenges.unify.id/v1/mle/sample_test.json'
    with request.urlopen(test_url) as response:
        if response.getcode() == 200:
            source = response.read()
            data = json.loads(source)
            test_df, user_stats, invalid_idx = validateFeaturizeData(data['attempts'], 'test', col_names[:-1])

    final_idx = []
    for i, idx in enumerate(invalid_idx):
        final_idx.append(idx - i)

    # predicting for the test data
    X_test = test_df.iloc[:, 1:-1]
    # pred_class = model.predict_classes(X_test)
    pred_class = np.argmax(model.predict(X_test), axis=-1)
    # print(pred_class.shape)
    pred_class = np.insert(pred_class, final_idx, -1)
    pred_user = []
    for user in pred_class:
        pred_user.append(mapping[user])


def readCommand( argv ):
    """
    Processes the command used to run pacman from the command line.
    """
    from optparse import OptionParser
    usageStr = """
    USAGE:      python run.py <options>
    EXAMPLES:   (1) python run.py
                  - starts in training mode
              (2) python run.py --test --''
              #TODO to be completed
    """
    parser = OptionParser(usageStr)

    parser.add_option('-m', '--mode', dest='mode',
                    help=default('Chose either one of the two modes - train, test'),
                    metavar='mode', default='train')
    parser.add_option('-url', dest='url',
                    help=default('the url to load data from'),metavar='URL',
                    default='https://challenges.unify.id/v1/mle/user_4a438fdede4e11e9b986acde48001122.json')

if __name__ == '__main__':
    args = readCommand(sys.argv[1:])
    no_features = len('Be Authentic. Be Yourself. Be Typing.')
    col_names = ['f_{}'.format(i) for i in range(1, no_features + 1)]
    col_names += ['del', 'user_id']
    checkpoint_filepath = os.path.join(os.getcwd(), 'model_checkpoint/')