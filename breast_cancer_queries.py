from sklearn.tree import DecisionTreeClassifier
import time
import csv
import sys
from hle import high_level_single # our code

with open('data/breast-cancer.csv', 'r') as f:
    reader = csv.reader(f, delimiter=';')
    full_dataset = list(reader)

features = {
    'clumpThickness': 'numeric',
    'uniformityCellSize': 'numeric',
    'uniformityCellShape': 'numeric',
    'marginalAdhesion': 'numeric',
    'singleEpiCellSize': 'numeric',
    'bareNuclei': 'numeric',
    'blandChromatin': 'numeric',
    'normalNucleoli':'numeric',
    'mitoses': 'numeric',
}

class_names = ['benign', 'melignant']

feature_names = list(features.keys())
feature_types = list(features.values())

# because of binary features with values that are not 0 or 1.
feature_mapping = {

}

def process_features_student(row):
    to_delete = [0]
    cpy = []
    for i in range(len(row)):
        if i not in to_delete:
            value = row[i]
            if value in feature_mapping:
                cpy.append(feature_mapping[value])
            else:
                cpy.append(value)
    assert len(cpy) == len(feature_names)
    return cpy

def process_class(val):
    if float(val) >= 3: # good grade is a grade in  [10, 20]. Bad grade is [0, 10)
        return 0
    else:
        return 1

dataset = full_dataset[1:]
X = [ process_features_student(data[:-1]) for data in dataset]
y = [ process_class(data[-1]) for data in dataset]

cancer_clf = DecisionTreeClassifier(max_leaf_nodes=400, random_state=0)
cancer_clf.fit(X, y)

print('DecisionTreeClassifier has been trained')

q1 = 'exists p1, exists p2, benign(p1) implies benign(p2)'
q2 = 'exists p1, exists p2, p1.blandChromatin > 3 and p2.marginalAdhesion <= 3 and melignant(p1) implies benign(p2)'
q3 = 'for every patient, patient.blandChromatin > 4 implies melignant(patient)'
q4 = ('exists p1, exists p2, exists p3, p1.marginalAdhesion > 8 and p2.bareNuclei <= 4'
      ' and p3.blandChromatin > 5 and p1.bareNuclei > 5 implies p2.bareNuclei <= 4'
      ' and p3.singleEpiCellSize > 2 implies benign(p2)')
q5 = ('for every p1, p1.marginalAdhesion > 8 implies p1.bareNuclei <= 4'
      ' and p1.bareNuclei > 5 implies p1.bareNuclei <= 4')
q6 = ('exists p1, exists p2, p1.mitoses <= 2 implies melignant(p1)'
      'and p2.blandChromatin > 9 implies p1.blandChromatin <= 3')



def example_queries():
    queries = [q1,q2,q3,q4,q5,q6]
    avg = 0
    for iq, query in enumerate(queries):
        t1 = time.perf_counter()
        answer = high_level_single(cancer_clf, feature_names, feature_types, class_names, query)[:-1]
        delta = time.perf_counter() - t1
        avg = avg + delta
        print(f'q{iq+1}: answer={answer}, time={delta}')
    avg = avg / (len(queries))
    print(f'average time = {avg}')

def query_from_file(filename):
    with open(filename, 'r') as f:
        query = f.read()
        query = ' '.join(query.replace('\n','').split())
        t1 = time.perf_counter()
        answer = high_level_single(cancer_clf, feature_names, feature_types, class_names, query)[:-1]
        delta = time.perf_counter() - t1
        print(f'answer={answer}, time={delta}')

if len(sys.argv) > 2:
    assert sys.argv[1] == '--query'
    filename = sys.argv[2]
    print(f'Evaluating query from file {filename}...')
    query_from_file(filename)
else:
    example_queries()
