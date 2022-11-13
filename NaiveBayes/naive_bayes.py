import operator


def read_stdin():
    training_data = []
    test_data = []
    count = 0
    classes = set()
    while True:
        try:
            line = input()
            if not line:
                break
            else:
                item = line.split(',')
                if item[-1] != '-1':
                    if count == 0:
                        features = [_ for _ in item]
                        count += 1
                    else:
                        training_data.append([_ for _ in item])
                        classes.add(item[-1])
                else:
                    test_data.append([_ for _ in item])
            count += 1
        except EOFError:
            break
    return training_data, test_data, list(classes), features


def get_class_count(training_data: dict):
    class_count = dict()
    for item in training_data:
        label = item[-1]
        if label not in class_count.keys():
            class_count[label] = 0
        class_count[label] += 1
    return class_count


def class_probability(training_data: list) -> dict:
    class_prob = dict()
    total = len(training_data)
    number_of_classes = len(set([i[-1] for i in training_data]))
    class_count = get_class_count(training_data)
    for k, v in class_count.items():
        if k not in class_count.keys():
            class_prob[k] = 0
        class_prob[k] = (v + 0.1) / (total + (0.1 * number_of_classes))
    return class_count, class_prob


def reformat_data(features, training_data: list) -> dict:
    data = list(map(list, zip(*training_data)))
    data = dict(zip(features, data))
    return data


def get_feature_values(training_data, features):
    feature_value = dict()
    for index, values in enumerate(training_data):
        data_item = list(zip(features, values))
        for feature in data_item[:-1]:
            if feature[0] not in feature_value.keys():
                feature_value[feature[0]] = set()
            feature_value[feature[0]].add(feature[1])
    return feature_value


def get_class_feature_count(training_data, features):
    class_feature_count = dict()
    for index, values in enumerate(training_data):
        data_item = list(zip(features, values))
        class_type = values[-1]
        if class_type not in class_feature_count.keys():
            class_feature_count[class_type] = {}
        for feature in data_item:
            if feature[0] not in class_feature_count[class_type].keys():
                class_feature_count[class_type][feature[0]] = {}
                for value in list(features[feature[0]]):
                    class_feature_count[class_type][feature[0]][value] = 0
            class_feature_count[class_type][feature[0]][feature[1]] += 1
    return class_feature_count


def calculate_feature_prob(class_count: dict, feature_count: dict):
    for label, l_values in feature_count.items():
        label_count = class_count[label]
        for feature, f_values in l_values.items():
            f_count = len(f_values)
            for value, count in f_values.items():
                f_values[value] = (count + 0.1) / (label_count + 0.1 * f_count)
    return feature_count


def predict(test_set: list, feature_prob: dict, features):
    class_predictions = []
    class_prob = dict()
    for test_item in test_set:
        data_item = list(zip(features, test_item))
        for label, l_values in feature_prob.items():
            prob = 1
            for data in data_item[1:-1]:
                test_feature = data[0]
                test_feature_value = data[1]
                if label not in class_prob.keys():
                    class_prob[label] = 0
                if test_feature_value in feature_prob[label][test_feature].keys():
                    prob *= feature_prob[label][test_feature][test_feature_value]
            class_prob[label] = prob
        print(max(class_prob.items(), key=operator.itemgetter(1))[0])
    return


training_data, test_data, classes, features = read_stdin()
training_count = len(training_data)
class_count, class_prob = class_probability(training_data)
feature_values = get_feature_values(training_data, features)
feature_count = get_class_feature_count(training_data, feature_values)
feature_prob = calculate_feature_prob(class_count, feature_count)

predict(test_data, feature_prob, features)