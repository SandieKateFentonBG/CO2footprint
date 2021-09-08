import numpy as np

x_features_str = ['Sector', 'Type', 'Basement', 'Foundations', 'Ground Floor', 'Superstructure', 'Cladding',
                  'BREEAM Rating']
x_features_int = ['GIFA (m2)', 'Storeys', 'Typical Span (m)', 'Typ Qk (kN_per_m2)']

all_features = x_features_str + x_features_int
GIFA_power=[1, 2, 3]
Storeys_power=[1, 2, 3]
Span_power=[]
Qk_power = [1, 2, 3]
Sector_power = [1]
Type_power = [1]
Basement_power=[]
Foundations_power=[]
Groundfloor_power=[]
Superstructure_power=[]
Cladding_power=[1]
Rating_power=[1]

print("here", np.power(GIFA_power, Span_power))

if Foundations_power:
    print("a", len(Foundations_power))

if not Foundations_power:
    print("b", len(Foundations_power))
print("a")
all_powers = [Sector_power, Type_power, Basement_power, Foundations_power, Groundfloor_power, Superstructure_power,
          Cladding_power, Rating_power,  GIFA_power,Storeys_power,Span_power,Qk_power]


def longest(list):
    return max(len(elem) for elem in list)

selected_features = []
selected_powers = []
max_power = longest(all_powers)
dictio = dict()
for feature, power_list in zip(all_features, all_powers):
    if power_list:
        dictio[feature] = power_list + [0] * (max_power - len(power_list))
        selected_features.append(feature)
        selected_powers.append(power_list + [0] * (max_power - len(power_list)))
myvalues = [2]*len(selected_features)
print("dictio", dictio)
print(selected_features),
print(selected_powers)
print(myvalues)
print(selected_powers[:, 1])

"""def power_up_feature(featureArray_column, selected_powers, max_power):
    list_of_virtual_features = []
    for i in range(max_power):

        for feature in feature-list :

    power in selected_powers:
        list_of_virtual_features.append(np.power(featureArray_column, selected_powers[:, i]).reshape([-1, 1]))
    return np.hstack(list_of_virtual_features)"""


v_f = np.power(myvalues, selected_powers).reshape([-1, 1])
print(v_f)
"""
print([0]*5)
r = [1, 2, 3]
j = [0]*5
print(r+j)
base = np.ones((5, 1))
add = np.zeros((5, 2))
print(base)
print(add)
all = np.concatenate((base, add), axis = 1)


print(all)

base = np.ones((5, 1))
print(base)
s = 1
if s:
    print("a")
a = 3
print(list(range(1, a+1)))

data = np.zeros((5, 3))
for i in range(5):  # 80
    data[i, :] = np.array(range(3))

b = np.array(range(3))
print(b)

print(data)
print(data.shape[0])"""
print(data.shape[1])