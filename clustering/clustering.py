import pandas as pd
import utils
import matplotlib.pyplot as plt

data = pd.read_csv('Mall_Customers.csv')

#cechy ktore chcemy przyrownac
annual_income = data['Annual Income (k$)']
spending = data['Spending Score (1-100)']
df = pd.concat([annual_income, spending], axis=1)

#wyswietlamy dane
plt.scatter(annual_income, spending)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

data = list(zip(df["Annual Income (k$)"], df["Spending Score (1-100)"]))

#metoda lokcia (zeby wiedziec ile k=ilosc_klastrow dobrac)
#inertia = wcss = within cluster sum of squares


colors = ["red", "blue", "green", "purple", "yellow"]
loss_list = [[] for _ in range(6)]

for i in range(10):
    for j in range(1, 7):
        kmeans = utils.Kmeans(j)
        clusters, labels, loss = kmeans.fit(data)
        loss_list[j - 1].append(loss)

loss_list = [sum(loss) / 10 for loss in loss_list]
plt.plot(range(1, 7), loss_list)
plt.show()

kmeans = utils.Kmeans(5)
clusters, labels, loss = kmeans.fit(data)

for i in range(len(labels)):
    class_xs = [labels[i][j][0] for j in range(len(labels[i]))]
    class_ys = [labels[i][j][1] for j in range(len(labels[i]))]

    plt.scatter(class_xs, class_ys, c=colors[i])
plt.show()